import base64
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types
from langchain_b12.genai.genai_utils import (
    convert_base_message_to_parts,
    convert_messages_to_contents,
    multi_content_to_part,
    parse_response_candidate,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class TestMultiContentToPart:
    """Test the multi_content_to_part function."""

    def test_text_content(self):
        """Test converting text content to Part."""
        contents = [{"type": "text", "text": "Hello world"}]
        parts = multi_content_to_part(contents)

        assert len(parts) == 1
        assert parts[0].text == "Hello world"

    def test_empty_text_content(self):
        """Test converting empty text content to Part."""
        contents = [{"type": "text", "text": ""}]
        parts = multi_content_to_part(contents)

        assert len(parts) == 0

    def test_image_url_content(self):
        """Test converting image_url content to Part."""
        # Create a simple base64 encoded image
        image_data = b"fake image data"
        encoded_data = base64.b64encode(image_data).decode("utf-8")

        contents = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_data}"},
            }
        ]

        with patch.object(types.Part, "from_bytes") as mock_from_bytes:
            mock_part = MagicMock()
            mock_from_bytes.return_value = mock_part

            parts = multi_content_to_part(contents)

            assert len(parts) == 1
            mock_from_bytes.assert_called_once_with(
                data=image_data, mime_type="image/jpeg"
            )

    def test_image_content_with_base64_data(self):
        """Test converting image content with base64 data to Part."""
        image_data = b"fake image data"
        encoded_data = base64.b64encode(image_data).decode("utf-8")

        contents = [{"type": "image", "data": encoded_data, "mime_type": "image/png"}]

        with patch.object(types.Part, "from_bytes") as mock_from_bytes:
            mock_part = MagicMock()
            mock_from_bytes.return_value = mock_part

            parts = multi_content_to_part(contents)

            assert len(parts) == 1
            mock_from_bytes.assert_called_once_with(
                data=image_data, mime_type="image/png"
            )

    def test_image_content_with_url(self):
        """Test converting image content with URL to Part."""
        contents = [
            {
                "type": "image",
                "url": "https://example.com/image.jpg",
                "mime_type": "image/jpeg",
            }
        ]

        with patch.object(types.Part, "from_uri") as mock_from_uri:
            mock_part = MagicMock()
            mock_from_uri.return_value = mock_part

            parts = multi_content_to_part(contents)

            assert len(parts) == 1
            mock_from_uri.assert_called_once_with(
                file_uri="https://example.com/image.jpg", mime_type="image/jpeg"
            )

    def test_image_content_with_url_no_mime_type(self):
        """Test converting image content with URL but no mime_type to Part."""
        contents = [{"type": "image", "url": "https://example.com/image.jpg"}]

        with patch.object(types.Part, "from_uri") as mock_from_uri:
            mock_part = MagicMock()
            mock_from_uri.return_value = mock_part

            parts = multi_content_to_part(contents)

            assert len(parts) == 1
            mock_from_uri.assert_called_once_with(
                file_uri="https://example.com/image.jpg", mime_type=None
            )

    def test_file_content_with_base64_data(self):
        """Test converting file content with base64 data to Part."""
        file_data = b"fake file data"
        encoded_data = base64.b64encode(file_data).decode("utf-8")

        contents = [
            {"type": "file", "data": encoded_data, "mime_type": "application/pdf"}
        ]

        with patch.object(types.Part, "from_bytes") as mock_from_bytes:
            mock_part = MagicMock()
            mock_from_bytes.return_value = mock_part

            parts = multi_content_to_part(contents)

            assert len(parts) == 1
            mock_from_bytes.assert_called_once_with(
                data=file_data, mime_type="application/pdf"
            )

    def test_file_content_with_url(self):
        """Test converting file content with URL to Part."""
        contents = [
            {
                "type": "file",
                "url": "https://example.com/document.pdf",
                "mime_type": "application/pdf",
            }
        ]

        with patch.object(types.Part, "from_uri") as mock_from_uri:
            mock_part = MagicMock()
            mock_from_uri.return_value = mock_part

            parts = multi_content_to_part(contents)

            assert len(parts) == 1
            mock_from_uri.assert_called_once_with(
                file_uri="https://example.com/document.pdf", mime_type="application/pdf"
            )

    def test_invalid_content_type(self):
        """Test handling of invalid content type."""
        contents = [{"type": "invalid", "data": "test"}]

        with pytest.raises(ValueError, match="Unknown content type: invalid"):
            multi_content_to_part(contents)

    def test_missing_type_field(self):
        """Test handling of content without type field."""
        contents = [{"data": "test"}]

        with pytest.raises(AssertionError, match="Received dict content without type"):
            multi_content_to_part(contents)

    def test_non_dict_content(self):
        """Test handling of non-dict content."""
        contents = ["invalid"]

        with pytest.raises(AssertionError, match="Expected dict content"):
            multi_content_to_part(contents)

    def test_image_missing_data_and_url(self):
        """Test image content missing both data and url."""
        contents = [{"type": "image", "mime_type": "image/jpeg"}]

        with pytest.raises(
            ValueError,
            match="Expected either 'data' or 'url' in content for image type",
        ):
            multi_content_to_part(contents)

    def test_file_missing_data_and_url(self):
        """Test file content missing both data and url."""
        contents = [{"type": "file", "mime_type": "application/pdf"}]

        with pytest.raises(
            ValueError, match="Expected either 'data' or 'url' in content for file type"
        ):
            multi_content_to_part(contents)


class TestConvertBaseMessageToParts:
    """Test the convert_base_message_to_parts function."""

    def test_string_content(self):
        """Test converting message with string content."""
        message = HumanMessage(content="Hello world")
        parts = convert_base_message_to_parts(message)

        assert len(parts) == 1
        assert parts[0].text == "Hello world"

    def test_empty_string_content(self):
        """Test converting message with empty string content."""
        message = HumanMessage(content="")
        parts = convert_base_message_to_parts(message)

        assert len(parts) == 0

    def test_list_content(self):
        """Test converting message with list content."""
        content = [{"type": "text", "text": "Hello world"}]
        message = HumanMessage(content=content)

        parts = convert_base_message_to_parts(message)

        assert len(parts) == 1
        assert parts[0].text == "Hello world"

    def test_invalid_content_type(self):
        """Test handling of invalid content type."""
        # Create a mock message with invalid content type
        mock_message = MagicMock()
        mock_message.content = 123  # Invalid content type

        with pytest.raises(ValueError, match="Received unexpected content type"):
            convert_base_message_to_parts(mock_message)


class TestConvertMessagesToContents:
    """Test the convert_messages_to_contents function."""

    def test_human_message(self):
        """Test converting HumanMessage."""
        messages = [HumanMessage(content="Hello")]
        contents = convert_messages_to_contents(messages)

        assert len(contents) == 1
        assert isinstance(contents[0], types.UserContent)
        assert len(contents[0].parts) == 1
        assert contents[0].parts[0].text == "Hello"

    def test_ai_message_without_tool_calls(self):
        """Test converting AIMessage without tool calls."""
        messages = [AIMessage(content="Hi there")]
        contents = convert_messages_to_contents(messages)

        assert len(contents) == 1
        assert isinstance(contents[0], types.ModelContent)
        assert len(contents[0].parts) == 1
        assert contents[0].parts[0].text == "Hi there"

    def test_ai_message_with_tool_calls(self):
        """Test converting AIMessage with tool calls."""
        tool_calls = [{"name": "search", "args": {"query": "test"}, "id": "call_123"}]
        messages = [AIMessage(content="Let me search", tool_calls=tool_calls)]
        contents = convert_messages_to_contents(messages)

        assert len(contents) == 1
        assert isinstance(contents[0], types.ModelContent)
        assert len(contents[0].parts) == 2  # text + function call
        assert contents[0].parts[0].text == "Let me search"
        assert contents[0].parts[1].function_call.name == "search"
        assert contents[0].parts[1].function_call.args == {"query": "test"}
        assert contents[0].parts[1].function_call.id == "call_123"

    def test_tool_message_new_content(self):
        """Test converting ToolMessage that creates new content."""
        # First add an AI message
        messages = [
            HumanMessage(content="Search for test"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "test"}, "id": "call_123"}
                ],
            ),
            ToolMessage(
                content="Search results", name="search", tool_call_id="call_123"
            ),
        ]
        contents = convert_messages_to_contents(messages)

        assert len(contents) == 3  # tool calls and response are combined
        assert isinstance(contents[0], types.UserContent)
        assert isinstance(contents[1], types.ModelContent)
        assert isinstance(contents[2], types.UserContent)
        assert len(contents[0].parts) == 1
        assert len(contents[1].parts) == 1
        assert len(contents[2].parts) == 1
        assert contents[0].parts[0].text == "Search for test"
        assert contents[1].parts[0].text is None
        assert contents[1].parts[0].function_call.name == "search"
        assert contents[1].parts[0].function_call.args == {"query": "test"}
        assert contents[1].parts[0].function_call.id == "call_123"
        assert contents[2].parts[0].function_response.name == "search"
        assert contents[2].parts[0].function_response.id == "call_123"
        assert contents[2].parts[0].function_response.response == {
            "output": "Search results"
        }

    def test_multiple_tool_messages(self):
        """Test converting multiple ToolMessages."""
        # First add an AI message
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "test"}, "id": "call_123"},
                    {"name": "search", "args": {"query": "another"}, "id": "call_456"},
                ],
            ),
            ToolMessage(
                content="Search results", name="search", tool_call_id="call_123"
            ),
            ToolMessage(
                content="Another result", name="search", tool_call_id="call_456"
            ),
        ]
        contents = convert_messages_to_contents(messages)

        assert len(contents) == 2  # multiple tool messages create one content
        assert isinstance(contents[0], types.ModelContent)
        assert len(contents[0].parts) == 2
        assert contents[0].parts[0].function_call.name == "search"
        assert contents[0].parts[0].function_call.id == "call_123"
        assert contents[0].parts[0].function_call.args == {"query": "test"}
        assert contents[0].parts[1].function_call.name == "search"
        assert contents[0].parts[1].function_call.id == "call_456"
        assert contents[0].parts[1].function_call.args == {"query": "another"}
        assert isinstance(contents[1], types.UserContent)
        assert len(contents[1].parts) == 2
        assert contents[1].parts[0].function_response.name == "search"
        assert contents[1].parts[0].function_response.id == "call_123"
        assert contents[1].parts[0].function_response.response == {
            "output": "Search results"
        }
        assert contents[1].parts[1].function_response.name == "search"
        assert contents[1].parts[1].function_response.id == "call_456"
        assert contents[1].parts[1].function_response.response == {
            "output": "Another result"
        }

    def test_system_message_ignored(self):
        """Test that SystemMessage is ignored."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello"),
        ]
        contents = convert_messages_to_contents(messages)

        assert len(contents) == 1  # Only HumanMessage converted
        assert isinstance(contents[0], types.UserContent)

    def test_invalid_message_type(self):
        """Test handling of invalid message type."""
        # Create a mock message with invalid type
        invalid_message = MagicMock()
        invalid_message.__class__.__name__ = "InvalidMessage"

        messages = [invalid_message]

        with pytest.raises(ValueError, match="Invalid message type"):
            convert_messages_to_contents(messages)

    def test_tool_message_assertions(self):
        """Test ToolMessage assertions."""
        # Test missing name
        with pytest.raises(AssertionError, match="Tool name is required"):
            messages = [
                ToolMessage(content="result", name=None, tool_call_id="call_123")
            ]
            convert_messages_to_contents(messages)

        # Test non-string content
        with pytest.raises(AssertionError, match="Expected str content"):
            messages = [
                ToolMessage(content=["result"], name="search", tool_call_id="call_123")
            ]
            convert_messages_to_contents(messages)

    def test_ai_message_tool_call_assertions(self):
        """Test AIMessage tool call assertions."""
        # Test missing tool call ID
        with pytest.raises(AssertionError, match="Tool call ID is required"):
            tool_calls = [{"name": "search", "args": {"query": "test"}, "id": ""}]
            messages = [AIMessage(content="Let me search", tool_calls=tool_calls)]
            convert_messages_to_contents(messages)


class TestParseResponseCandidate:
    """Test the parse_response_candidate function."""

    def test_text_only_response(self):
        """Test parsing response with only text."""
        candidate = types.Candidate(
            content=types.Content(parts=[types.Part(text="Hello world")])
        )

        message = parse_response_candidate(candidate)

        assert message.content == "Hello world"
        assert message.additional_kwargs == {}
        assert message.tool_call_chunks == []

    def test_multiple_text_parts(self):
        """Test parsing response with multiple text parts."""
        candidate = types.Candidate(
            content=types.Content(
                parts=[types.Part(text="Hello"), types.Part(text=" world")]
            )
        )

        message = parse_response_candidate(candidate)

        assert message.content == ["Hello", " world"]

    def test_empty_content(self):
        """Test parsing response with no text content."""
        # Create a mock part with no text content
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        message = parse_response_candidate(mock_candidate)

        assert message.content == ""

    def test_function_call_response(self):
        """Test parsing response with function call."""
        # Create mock function call
        mock_function_call = MagicMock()
        mock_function_call.name = "search"
        mock_function_call.args = {"query": "test"}

        # Create mock part with function call
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call

        # Create mock content and candidate
        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = "generated_id"

            message = parse_response_candidate(mock_candidate)

        assert message.content == ""
        assert "function_call" in message.additional_kwargs
        expected_name = "search"
        assert message.additional_kwargs["function_call"]["name"] == expected_name
        expected_args = '{"query": "test"}'
        assert message.additional_kwargs["function_call"]["arguments"] == expected_args
        assert len(message.tool_call_chunks) == 1
        assert message.tool_call_chunks[0]["name"] == "search"
        assert message.tool_call_chunks[0]["args"] == '{"query": "test"}'

    def test_text_and_function_call_response(self):
        """Test parsing response with both text and function call."""
        # Create mock function call
        mock_function_call = MagicMock()
        mock_function_call.name = "search"
        mock_function_call.args = {"query": "test"}

        # Create real text part
        text_part = types.Part(text="Let me search for that")

        # Create mock function part
        mock_function_part = MagicMock()
        mock_function_part.text = None
        mock_function_part.function_call = mock_function_call

        # Create mock content and candidate
        mock_content = MagicMock()
        mock_content.parts = [text_part, mock_function_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        message = parse_response_candidate(mock_candidate)

        assert message.content == "Let me search for that"
        assert "function_call" in message.additional_kwargs

    def test_none_content_assertion(self):
        """Test assertion when candidate content is None."""
        candidate = types.Candidate(content=None)

        with pytest.raises(AssertionError, match="Response candidate content is None"):
            parse_response_candidate(candidate)

    def test_none_parts(self):
        """Test handling when parts is None."""
        mock_content = MagicMock()
        mock_content.parts = None

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        message = parse_response_candidate(mock_candidate)

        assert message.content == ""
        assert message.additional_kwargs == {}
        assert message.tool_call_chunks == []
