import base64
import json
import uuid
from collections.abc import Sequence
from typing import Any, cast

from google.genai import types
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk


def multi_content_to_part(
    contents: Sequence[dict[str, str | dict[str, str]] | str],
) -> list[types.Part]:
    """Convert sequence content to a Part object.

    Args:
        contents: A sequence of dictionaries representing content. Examples:
            [
                {
                    "type": "text",
                    "text": "This is a text message"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_artifact}"
                    },
                },
                {
                    "type": "file",
                    "file": {
                        "uri": f"gs://{bucket_name}/{file_name}",
                        "mime_type": mime_type,
                    }
                }
            ]
    """
    parts = []
    for content in contents:
        assert isinstance(content, dict), "Expected dict content"
        assert "type" in content, "Received dict content without type"
        if content["type"] == "text":
            assert "text" in content, "Expected 'text' in content"
            if content["text"]:
                assert isinstance(content["text"], str), "Expected str content"
                parts.append(types.Part(text=content["text"]))
        elif content["type"] == "image_url":
            assert isinstance(content["image_url"], dict), "Expected dict image_url"
            assert "url" in content["image_url"], "Expected 'url' in content"
            split_url: tuple[str, str] = content["image_url"]["url"].split(",", 1)  # type: ignore
            header, encoded_data = split_url
            mime_type = header.split(":", 1)[1].split(";", 1)[0]
            data = base64.b64decode(encoded_data)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        elif content["type"] == "file":
            assert "file" in content, "Expected 'file' in content"
            file = content["file"]
            assert isinstance(file, dict), "Expected dict file"
            assert "uri" in file, "Expected 'uri' in content['file']"
            assert "mime_type" in file, "Expected 'mime_type' in content['file']"
            parts.append(
                types.Part.from_uri(file_uri=file["uri"], mime_type=file["mime_type"])
            )
        else:
            raise ValueError(f"Unknown content type: {content['type']}")
    return parts


def convert_base_message_to_parts(
    message: BaseMessage,
) -> list[types.Part]:
    """Convert a LangChain BaseMessage to Google GenAI Content object."""
    parts = []
    if isinstance(message.content, str):
        if message.content:
            parts.append(types.Part(text=message.content))
    elif isinstance(message.content, list):
        parts.extend(multi_content_to_part(message.content))
    else:
        raise ValueError(
            "Received unexpected content type, "
            f"expected str or list, but got {type(message.content)}"
        )
    return parts


def convert_messages_to_contents(
    messages: Sequence[BaseMessage],
) -> list[types.Content]:
    """Convert a sequence of LangChain messages to Google GenAI Content objects.

    Args:
        messages: A sequence of LangChain BaseMessage objects

    Returns:
        A list of Google GenAI Content objects
    """
    contents = []

    for message in messages:
        if isinstance(message, HumanMessage):
            parts = convert_base_message_to_parts(message)
            contents.append(types.UserContent(parts=parts))
        elif isinstance(message, AIMessage):
            text_parts = convert_base_message_to_parts(message)
            function_parts = []
            if message.tool_calls:
                # Example of tool_call
                # tool_call = {
                #     "name": "foo",
                #     "args": {"a": 1},
                #     "id": "123"
                # }
                for tool_call in message.tool_calls:
                    tool_id = tool_call["id"]
                    assert tool_id, "Tool call ID is required"
                    function_parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tool_call["name"],
                                args=tool_call["args"],
                                id=tool_id,
                            ),
                        )
                    )

            contents.append(
                types.ModelContent(
                    parts=[*text_parts, *function_parts],
                )
            )
        elif isinstance(message, ToolMessage):
            # Note: We tried combining function_call and function_response into one
            # part, but that throws a 4xx server error.
            assert isinstance(message.content, str), "Expected str content"
            assert message.name, "Tool name is required"
            tool_part = types.Part(
                function_response=types.FunctionResponse(
                    id=message.tool_call_id,
                    name=message.name,
                    response={"output": message.content},
                ),
            )

            # Ensure that all function_responses are in a single content
            last_content = contents[-1]
            last_content_part = last_content.parts[-1]
            if last_content_part.function_response:
                # Merge with the last content
                last_content.parts.append(tool_part)
            else:
                # Create a new content
                contents.append(types.UserContent(parts=[tool_part]))
        elif isinstance(message, SystemMessage):
            # There is no genai.types equivalent for SystemMessage
            pass
        else:
            raise ValueError(f"Invalid message type: {type(message)}")

    return contents


def parse_response_candidate(response_candidate: types.Candidate) -> AIMessageChunk:
    content: None | str | list[str] = None
    additional_kwargs = {}
    tool_call_chunks = []

    assert response_candidate.content, "Response candidate content is None"
    for part in response_candidate.content.parts or []:
        try:
            text: str | None = part.text
        except AttributeError:
            text = None

        if text:
            if not content:
                content = text
            elif isinstance(content, str):
                content = [content, text]
            elif isinstance(content, list):
                content.append(text)
            else:
                raise ValueError("Unexpected content type")

        if part.function_call:
            # For backward compatibility we store a function call in additional_kwargs,
            # but in general the full set of function calls is stored in tool_calls.
            function_call = {"name": part.function_call.name}
            # dump to match other function calling llm for now
            function_call_args_dict = part.function_call.args
            assert function_call_args_dict is not None
            function_call["arguments"] = json.dumps(function_call_args_dict)
            additional_kwargs["function_call"] = function_call

            index = function_call.get("index")
            tool_call_chunks.append(
                tool_call_chunk(
                    name=function_call.get("name"),
                    args=function_call.get("arguments"),
                    id=function_call.get("id", str(uuid.uuid4())),
                    index=int(index) if index else None,
                )
            )
    if content is None:
        content = ""

    return AIMessageChunk(
        content=cast(str | list[str | dict[Any, Any]], content),
        additional_kwargs=additional_kwargs,
        tool_call_chunks=tool_call_chunks,
    )
