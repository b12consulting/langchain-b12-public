import base64
from collections.abc import Sequence

from google.genai import types
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)


def multi_content_to_part(
    contents: Sequence[dict | str],
) -> list[types.Part]:
    """Convert image content to a Part object.

    Args:
        contents: A sequence of dictionaries representing image content. Examples:
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
                }
                {
                    "type": "file",
                    "uri": f"gs://{bucket_name}/{file_name}",
                    "mime_type": mime_type,
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
                parts.append(types.Part(text=content["text"]))
        elif content["type"] == "image_url":
            assert "url" in content["image_url"], "Expected 'url' in content"
            split_url: tuple[str, str] = content["image_url"]["url"].split(",", 1)
            header, encoded_data = split_url
            mime_type = header.split(":", 1)[1].split(";", 1)[0]
            data = base64.b64decode(encoded_data)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        elif content["type"] == "file":
            assert "uri" in content, "Expected 'uri' in content"
            parts.append(
                types.Part.from_uri(
                    file_uri=content["uri"],
                    mime_type=content["mime_type"],
                )
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
            # Google's documentation is seemingly incorrect about the content role
            # only being allowed as "model" or "user". It can be "function" as well.
            # We tried combining function_call and function_response into one part, but
            # that throws a 4xx server error.
            assert isinstance(message.content, str), "Expected str content"
            assert message.name, "Tool name is required"
            contents.append(
                types.Content(
                    role="function",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=message.tool_call_id,
                                name=message.name,
                                response={"output": message.content},
                            ),
                        )
                    ],
                )
            )
        else:
            raise ValueError(f"Invalid message type: {type(message)}")

    return contents
