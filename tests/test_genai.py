from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import _api_client as genai_api_client
from google.genai import Client, types
from google.genai.errors import ClientError
from langchain_b12.genai.genai import ChatGenAI
from langchain_core.messages import HumanMessage
from pydantic import ValidationError


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


def test_chatgenai_maps_max_retries_to_http_options():
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = _make_success_iter()
    model = ChatGenAI(client=client, max_retries=2)
    model.invoke([HumanMessage(content="foo")])

    config = client.models.generate_content_stream.call_args.kwargs["config"]
    assert config.http_options.retry_options.attempts == 3


def test_chatgenai_uses_custom_http_retry_options():
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = _make_success_iter()
    retry_options = types.HttpRetryOptions(attempts=7, initial_delay=0.25)
    model = ChatGenAI(client=client, http_retry_options=retry_options)
    model.invoke([HumanMessage(content="foo")])

    config = client.models.generate_content_stream.call_args.kwargs["config"]
    assert config.http_options.retry_options == retry_options


def test_chatgenai_rejects_multiple_constructor_retry_options():
    with pytest.raises(
        ValidationError,
        match="max_retries and http_retry_options cannot both be provided",
    ):
        ChatGenAI(
            client=MagicMock(spec=Client),
            max_retries=3,
            http_retry_options=types.HttpRetryOptions(attempts=4),
        )


def test_chatgenai_rejects_none_max_retries():
    with pytest.raises(ValidationError):
        ChatGenAI(client=MagicMock(spec=Client), max_retries=None)


def test_chatgenai_merges_per_call_http_options():
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = _make_success_iter()
    model = ChatGenAI(client=client, max_retries=2)
    model.invoke(
        [HumanMessage(content="foo")],
        http_options=types.HttpOptions(timeout=1_000),
    )

    config = client.models.generate_content_stream.call_args.kwargs["config"]
    assert config.http_options.timeout == 1_000
    assert config.http_options.retry_options.attempts == 3


def test_chatgenai_rejects_conflicting_per_call_retry_options():
    client: Client = MagicMock(spec=Client)
    model = ChatGenAI(client=client, max_retries=2)

    with pytest.raises(ValueError, match="Per-call retry_options cannot be combined"):
        model.invoke(
            [HumanMessage(content="foo")],
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=4)
            ),
        )


def test_chatgenai_allows_per_call_retry_options_when_defaults_are_implicit():
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = _make_success_iter()
    retry_options = types.HttpRetryOptions(attempts=4)
    model = ChatGenAI(client=client)
    model.invoke(
        [HumanMessage(content="foo")],
        http_options=types.HttpOptions(retry_options=retry_options),
    )

    config = client.models.generate_content_stream.call_args.kwargs["config"]
    assert config.http_options.retry_options == retry_options


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


def test_stream_no_retry_after_first_chunk():
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


def test_stream_does_not_retry_first_chunk_failure():
    client: Client = MagicMock(spec=Client)

    def fail_on_first_next():
        raise Exception("First chunk error")
        yield  # Make it a generator

    client.models.generate_content_stream.return_value = fail_on_first_next()

    model = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="foo")]

    with pytest.raises(Exception, match="First chunk error"):
        list(model.stream(messages))
    assert client.models.generate_content_stream.call_count == 1


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
async def test_astream_no_retry_after_first_chunk():
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


def _fake_429_error() -> ClientError:
    return ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "429 Rate limit exceeded",
            }
        },
        None,
    )


def _success_http_response() -> genai_api_client.HttpResponse:
    chunk = '{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]}}]}'
    return genai_api_client.HttpResponse(
        headers={"status-code": "200"},
        response_stream=[chunk],
    )


def test_google_genai_sync_retry(monkeypatch):
    call_count = 0

    def mock_request_once(self, http_request, stream=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _fake_429_error()
        return _success_http_response()

    monkeypatch.setattr(genai_api_client.BaseApiClient, "_request_once", mock_request_once)
    model = ChatGenAI(client=Client(api_key="fake"), max_retries=1)

    assert model.invoke([HumanMessage(content="hello")]).content == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_google_genai_async_retry(monkeypatch):
    call_count = 0

    async def mock_request_once(self, http_request, stream=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _fake_429_error()
        return _success_http_response()

    monkeypatch.setattr(
        genai_api_client.BaseApiClient,
        "_async_request_once",
        mock_request_once,
    )
    model = ChatGenAI(client=Client(api_key="fake"), max_retries=1)

    assert (await model.ainvoke([HumanMessage(content="hello")])).content == "ok"
    assert call_count == 2
