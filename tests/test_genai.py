from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import Client, types
from langchain_core.messages import HumanMessage

from langchain_b12.genai.genai import ChatGenAI


def _make_response_chunk(text: str) -> types.GenerateContentResponse:
    """Helper to create a response chunk."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(parts=[types.Part(text=text)]))
        ]
    )


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
            _make_response_chunk("bar"),
            _make_response_chunk("baz"),
        )
    )
    model = ChatGenAI(client=client)
    messages = [HumanMessage(content="foo")]
    response = model.invoke(messages)
    method: MagicMock = client.models.generate_content_stream
    method.assert_called_once()
    assert response.content == "barbaz"


def test_max_retries_converts_to_http_retry_options():
    """Test that max_retries is properly converted to HttpRetryOptions."""
    client = MagicMock(spec=Client)
    model = ChatGenAI(client=client, max_retries=5)

    assert model.http_retry_options is not None
    assert model.http_retry_options.attempts == 5


def test_http_retry_options_passed_directly():
    """Test that http_retry_options can be passed directly."""
    client = MagicMock(spec=Client)
    retry_options = types.HttpRetryOptions(
        attempts=10,
        initial_delay=2.0,
        max_delay=30.0,
    )
    model = ChatGenAI(client=client, http_retry_options=retry_options)

    assert model.http_retry_options == retry_options
    assert model.http_retry_options.attempts == 10
    assert model.http_retry_options.initial_delay == 2.0


def test_http_retry_options_overrides_max_retries():
    """Test that explicit http_retry_options overrides max_retries."""
    client = MagicMock(spec=Client)
    retry_options = types.HttpRetryOptions(attempts=7)
    model = ChatGenAI(client=client, max_retries=3, http_retry_options=retry_options)

    # http_retry_options should take precedence
    assert model.http_retry_options == retry_options
    assert model.http_retry_options.attempts == 7


def test_retry_options_passed_in_stream_config():
    """Test that retry options are passed to GenerateContentConfig."""
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = iter(
        [_make_response_chunk("success")]
    )

    model = ChatGenAI(client=client, max_retries=5)
    messages = [HumanMessage(content="foo")]
    response = model.invoke(messages)

    # Verify the config was called with http_options containing retry_options
    call_args = client.models.generate_content_stream.call_args
    config = call_args.kwargs["config"]
    assert config.http_options is not None
    assert config.http_options.retry_options is not None
    assert config.http_options.retry_options.attempts == 5


def test_no_retry_options_when_max_retries_none():
    """Test that no http_retry_options are set when max_retries is None."""
    client = MagicMock(spec=Client)
    model = ChatGenAI(client=client, max_retries=None)

    assert model.http_retry_options is None


# --- Streaming behavior tests ---


def test_stream_yields_chunks_immediately():
    """Test that stream yields chunks as they arrive, not buffered."""
    client: Client = MagicMock(spec=Client)
    chunks_yielded: list[str] = []

    def mock_stream():
        for text in ["chunk1", "chunk2", "chunk3"]:
            # Track when chunks are yielded from the source
            chunks_yielded.append(f"source:{text}")
            yield _make_response_chunk(text)

    client.models.generate_content_stream.return_value = mock_stream()

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    received: list[str] = []
    for chunk in model.stream(messages):
        received.append(chunk.content)
        # After receiving each chunk, check that source yielded it
        assert len(received) == len([c for c in chunks_yielded if c.startswith("source:")])

    assert received == ["chunk1", "chunk2", "chunk3"]


def test_stream_error_propagates():
    """Test that errors during streaming are propagated."""
    client: Client = MagicMock(spec=Client)

    def failing_stream():
        yield _make_response_chunk("first")
        raise Exception("Mid-stream error")

    client.models.generate_content_stream.return_value = failing_stream()

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    chunks = []
    with pytest.raises(Exception, match="Mid-stream error"):
        for chunk in model.stream(messages):
            chunks.append(chunk.content)

    # First chunk was received before error
    assert chunks == ["first"]


# --- Async streaming tests ---


async def _async_iter(items):
    """Helper to create an async iterator from items."""
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_astream_yields_chunks_immediately():
    """Test that async stream yields chunks as they arrive."""
    client: Client = MagicMock(spec=Client)

    chunks = [
        _make_response_chunk("async1"),
        _make_response_chunk("async2"),
        _make_response_chunk("async3"),
    ]

    # generate_content_stream returns a coroutine that resolves to async iterator
    client.aio.models.generate_content_stream = AsyncMock(
        return_value=_async_iter(chunks)
    )

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    received: list[str] = []
    async for chunk in model.astream(messages):
        received.append(chunk.content)

    assert received == ["async1", "async2", "async3"]


@pytest.mark.asyncio
async def test_astream_error_propagates():
    """Test that errors during async streaming are propagated."""
    client: Client = MagicMock(spec=Client)

    async def failing_after_first():
        yield _make_response_chunk("first")
        raise Exception("Async mid-stream error")

    client.aio.models.generate_content_stream = AsyncMock(
        return_value=failing_after_first()
    )

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    chunks = []
    with pytest.raises(Exception, match="Async mid-stream error"):
        async for chunk in model.astream(messages):
            chunks.append(chunk.content)

    assert chunks == ["first"]
