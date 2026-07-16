"""Test for empty stream handling in ChatGenAI."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import Client
from langchain_core.messages import HumanMessage

from langchain_b12.genai.genai import ChatGenAI


async def _empty_async_iter():
    """Helper to create an empty async iterator."""
    return
    yield  # Make this an async generator


def test_empty_stream_raises_langchain_aggregation_error():
    """Aggregating an empty sync stream uses LangChain's standard error."""
    # Create a mock client
    client: Client = MagicMock(spec=Client)
    client.models.generate_content_stream.return_value = iter([])  # Empty iterator

    # Pass the mock client directly
    llm = ChatGenAI(client=client)

    messages = [HumanMessage(content="test")]

    with pytest.raises(ValueError, match="No generations found in stream"):
        llm.invoke(messages)


@pytest.mark.asyncio
async def test_empty_async_stream_via_astream():
    """Public astream uses LangChain's standard empty-stream error."""
    # Create a mock client
    client: Client = MagicMock(spec=Client)

    # Mock async empty iterator
    client.aio.models.generate_content_stream = AsyncMock(
        return_value=_empty_async_iter()
    )

    # Pass the mock client directly
    llm = ChatGenAI(client=client)

    messages = [HumanMessage(content="test")]

    with pytest.raises(ValueError, match="No generation chunks were returned"):
        [chunk async for chunk in llm.astream(messages)]


@pytest.mark.asyncio
async def test_empty_async_stream_raises_langchain_aggregation_error():
    """Aggregating an empty async stream uses LangChain's standard error."""
    # Create a mock client
    client: Client = MagicMock(spec=Client)

    # Mock async empty iterator
    client.aio.models.generate_content_stream = AsyncMock(
        return_value=_empty_async_iter()
    )

    # Pass the mock client directly
    llm = ChatGenAI(client=client)

    messages = [HumanMessage(content="test")]

    with pytest.raises(ValueError, match="No generations found in stream"):
        await llm.ainvoke(messages)
