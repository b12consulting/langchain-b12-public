from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import Client, types
from langchain_b12.genai.genai import ChatGenAI
from langchain_core.messages import HumanMessage


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


def _make_success_iter():
    """Helper to create a successful streaming iterator."""
    return iter([_make_response_chunk("success")])


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
def test_chatgenai_retry_succeeds_after_failure(mock_wait):
    """Test that retry logic succeeds after transient failures."""
    client: Client = MagicMock(spec=Client)

    # First two calls fail, third succeeds
    client.models.generate_content_stream.side_effect = [
        Exception("Transient error 1"),
        Exception("Transient error 2"),
        _make_success_iter(),
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
    client.models.generate_content_stream.return_value = _make_success_iter()

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]
    response = model.invoke(messages)

    assert response.content == "success"
    assert client.models.generate_content_stream.call_count == 1


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


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
def test_stream_no_retry_after_first_chunk(mock_wait):
    """Test that errors after first chunk are NOT retried."""
    client: Client = MagicMock(spec=Client)

    def failing_after_first():
        yield _make_response_chunk("first")
        raise Exception("Mid-stream error")

    client.models.generate_content_stream.return_value = failing_after_first()

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    chunks = []
    with pytest.raises(Exception, match="Mid-stream error"):
        for chunk in model.stream(messages):
            chunks.append(chunk.content)

    # First chunk was received
    assert chunks == ["first"]
    # Only one call - no retry after first chunk
    assert client.models.generate_content_stream.call_count == 1


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
def test_stream_retry_on_first_chunk_failure(mock_wait):
    """Test that failure on first chunk triggers retry."""
    client: Client = MagicMock(spec=Client)

    def fail_on_first_next():
        raise Exception("First chunk error")
        yield  # Make it a generator

    def success_stream():
        yield _make_response_chunk("success1")
        yield _make_response_chunk("success2")

    client.models.generate_content_stream.side_effect = [
        fail_on_first_next(),
        success_stream(),
    ]

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    chunks = [chunk.content for chunk in model.stream(messages)]
    assert chunks == ["success1", "success2"]
    assert client.models.generate_content_stream.call_count == 2


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
@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
async def test_astream_no_retry_after_first_chunk(mock_wait):
    """Test that errors after first chunk are NOT retried in async."""
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
    assert client.aio.models.generate_content_stream.call_count == 1


@pytest.mark.asyncio
@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
async def test_astream_retry_succeeds_after_failure(mock_wait):
    """Test that async retry logic works for initial failures."""
    client: Client = MagicMock(spec=Client)

    call_count = 0

    async def side_effect_fn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Async transient error")
        return _async_iter([_make_response_chunk("async_success")])

    client.aio.models.generate_content_stream = AsyncMock(side_effect=side_effect_fn)

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    chunks = []
    async for chunk in model.astream(messages):
        chunks.append(chunk.content)

    assert chunks == ["async_success"]
    assert client.aio.models.generate_content_stream.call_count == 2
