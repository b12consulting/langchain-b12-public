from unittest.mock import MagicMock, patch

import pytest
from google.genai import Client, types
from langchain_b12.genai.genai import ChatGenAI
from langchain_core.messages import HumanMessage


def test_chatgenai():
    client = MagicMock(spec=Client)
    model = ChatGenAI(client=client, model="foo", temperature=1)
    assert model.model_name == "foo"
    assert model.temperature == 1
    assert model.client == client


def test_chatgenai_invocation():
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = iter(
        (
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text="bar")])
                    ),
                ]
            ),
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text="baz")])
                    ),
                ]
            ),
        )
    )
    model = ChatGenAI(client=client)
    messages = [HumanMessage(content="foo")]
    response = model.invoke(messages)
    method: MagicMock = client.models.generate_content_stream
    method.assert_called_once()
    assert response.content == "barbaz"


def _make_success_response():
    """Helper to create a successful streaming response."""
    return iter(
        [
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text="success")])
                    ),
                ]
            ),
        ]
    )


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
def test_chatgenai_retry_succeeds_after_failure(mock_wait):
    """Test that retry logic succeeds after transient failures."""
    client: Client = MagicMock(spec=Client)

    # First two calls fail, third succeeds
    client.models.generate_content_stream.side_effect = [
        Exception("Transient error 1"),
        Exception("Transient error 2"),
        _make_success_response(),
    ]

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]
    response = model.invoke(messages)

    assert response.content == "success"
    assert client.models.generate_content_stream.call_count == 3


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
def test_chatgenai_retry_exhausted_raises(mock_wait):
    """Test that exception is raised after all retries are exhausted."""
    client: Client = MagicMock(spec=Client)

    # All calls fail
    client.models.generate_content_stream.side_effect = Exception("Persistent error")

    model = ChatGenAI(client=client, max_retries=2)
    messages = [HumanMessage(content="foo")]

    with pytest.raises(Exception, match="Persistent error"):
        model.invoke(messages)

    # Initial attempt + 2 retries = 3 total calls
    assert client.models.generate_content_stream.call_count == 3


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
def test_chatgenai_no_retry_when_max_retries_zero(mock_wait):
    """Test that no retries occur when max_retries=0."""
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.side_effect = Exception("Error")

    model = ChatGenAI(client=client, max_retries=0)
    messages = [HumanMessage(content="foo")]

    with pytest.raises(Exception, match="Error"):
        model.invoke(messages)

    # Only 1 attempt, no retries
    assert client.models.generate_content_stream.call_count == 1


def test_chatgenai_no_retry_on_success():
    """Test that no retries occur when first attempt succeeds."""
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = _make_success_response()

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]
    response = model.invoke(messages)

    assert response.content == "success"
    assert client.models.generate_content_stream.call_count == 1
