"""Test for empty stream handling in ChatGenAI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import Client, types
from langchain_core.messages import HumanMessage

from langchain_b12.genai.genai import ChatGenAI


async def _empty_async_iter():
    """Helper to create an empty async iterator."""
    return
    yield  # Make this an async generator


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
def test_empty_stream_raises_clear_error(mock_wait):
    """Test that an empty stream from the API raises a clear error.

    This test verifies that when the API returns an empty stream,
    we raise a clear ValueError instead of a confusing RuntimeError
    due to PEP 479 StopIteration conversion.
    """
    # Create a mock client
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = iter([])  # Empty iterator

    # Pass the mock client directly
    llm = ChatGenAI(client=client)

    messages = [HumanMessage(content="test")]

    # After the fix, this should raise a clear ValueError
    with pytest.raises(ValueError, match="No response from model"):
        llm.invoke(messages)


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
@pytest.mark.asyncio
async def test_empty_async_stream_raises_clear_error(mock_wait):
    """Test that an empty async stream from the API raises a clear error.

    This reproduces the exact issue from the stack trace where _agenerate_with_cache
    collects empty chunks and then generate_from_stream raises "No generations found in stream".
    """
    # Create a mock client
    client: Client = MagicMock(spec=Client)

    # Mock async empty iterator
    client.aio.models.generate_content_stream = AsyncMock(
        return_value=_empty_async_iter()
    )

    # Pass the mock client directly
    llm = ChatGenAI(client=client)

    messages = [HumanMessage(content="test")]

    # This should raise a clear error, not "No generations found in stream"
    with pytest.raises(ValueError, match="No response from model"):
        await llm.ainvoke(messages)


@patch("langchain_b12.genai.genai.wait_exponential_jitter", return_value=lambda _: 0)
@pytest.mark.asyncio
async def test_async_stream_with_retry(mock_wait):
    """Test that async streaming works correctly after retries.

    This test ensures that after an initial failure (empty stream),
    the retry mechanism successfully recovers with a valid stream.
    """

    # Create a mock client
    client: Client = MagicMock(spec=Client)

    # Helper to create a response chunk
    def _make_response_chunk(text: str) -> types.GenerateContentResponse:
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(parts=[types.Part(text=text)]))
            ]
        )

    async def _success_async_iter():
        yield _make_response_chunk("success")

    # First call returns empty, second succeeds
    call_count = 0

    async def side_effect_fn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _empty_async_iter()
        return _success_async_iter()

    client.aio.models.generate_content_stream = AsyncMock(side_effect=side_effect_fn)

    llm = ChatGenAI(client=client, max_retries=3)
    messages = [HumanMessage(content="test")]

    # Should succeed after retry
    result = await llm.ainvoke(messages)
    assert result.content == "success"
    assert call_count == 2
