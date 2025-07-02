"""
Comprehensive tests for the CitationMixin class.
"""

from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from langchain_b12.citations.citations import Citation, CitationMixin, Citations
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MockChatModel(CitationMixin):
    response_content: str = Field(default="This is a mock response.")
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "simple"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):

        ai_message = AIMessage(
            content=self.response_content, tool_calls=self.tool_calls
        )
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])


class MockChatModelWithStructuredOutput(MockChatModel):
    structured_response: BaseModel = Field(...)

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return MockChatModel(
            response_content="",
            tool_calls=[
                ToolCall(
                    name=self.structured_response.__class__.__name__,
                    args=self.structured_response.model_dump(),
                    id="structured_abc",
                )
            ],
        )


class TestCitationMixin:
    """Test the CitationMixin class functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end(self):
        """Test that context tags are processed correctly."""

        citations = Citations(
            values=[Citation(sentence_index=0, key="abc", cited_text="bar")]
        )
        model = MockChatModelWithStructuredOutput(
            response_content="foo", structured_response=citations
        )

        # Create a message with context tags
        context_message = HumanMessage(
            content="Question about <context key='abc'>\nbaz\n</context>"
        )
        messages: list[BaseMessage] = [context_message]

        # Simulate structured content after citation processing
        expected_content = [
            {
                "text": "foo",
                "citations": [
                    {
                        "cited_text": "baz",
                        "generated_cited_text": "bar",
                        "key": "abc",
                        "dist": 1,
                    }
                ],
                "type": "text",
            }
        ]

        result = await model.ainvoke(messages)

        assert result.content == expected_content

    @pytest.mark.asyncio
    async def test_citation_mixin_basic_functionality_without_context(self):
        """Test basic functionality when no context tags are present."""

        model = MockChatModel(response_content="Test response.")
        messages: list[list[BaseMessage]] = [[HumanMessage(content="Test message")]]

        result = await model.agenerate(messages)

        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        generation = result.generations[0][0]
        assert isinstance(generation, ChatGeneration)
        # When no context tags, content remains string
        assert generation.message.content == "Test response."

    @pytest.mark.asyncio
    async def test_citation_mixin_basic_functionality_without_context_invoke(self):
        """Test basic functionality when no context tags are present."""

        model = MockChatModel(response_content="Test response.")
        messages: list[BaseMessage] = [HumanMessage(content="Test message")]

        result = await model.ainvoke(messages)

        assert isinstance(result, AIMessage)
        # When no context tags, content remains string
        assert result.content == "Test response."

    @pytest.mark.asyncio
    async def test_citation_mixin_recursion_prevention(self):
        """Test that CitationMixin prevents recursion when _adding_citations is True."""

        model = MockChatModel()

        messages: list[list[BaseMessage]] = [[HumanMessage(content="Test message")]]

        # Mock the parent's agenerate method
        with patch.object(
            CitationMixin.__bases__[0], "agenerate"
        ) as mock_parent_agenerate:
            mock_parent_agenerate.return_value = LLMResult(
                generations=[
                    [ChatGeneration(message=AIMessage(content="Test response"))]
                ]
            )

            result = await model.agenerate(messages)

            mock_parent_agenerate.assert_called_once()
            call_args, _ = mock_parent_agenerate.call_args
            assert call_args[0] == messages
            assert len(result.generations) == 1
            assert len(result.generations[0]) == 1

    @pytest.mark.asyncio
    async def test_citation_mixin_exception_handling(self):
        """Test that exceptions are handled properly."""

        class ErrorModel(CitationMixin):
            @property
            def _llm_type(self) -> str:
                return "error"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                raise ValueError("Test error")

        model = ErrorModel()
        messages: list[list[BaseMessage]] = [[HumanMessage(content="Test message")]]

        with pytest.raises(ValueError, match="Test error"):
            await model.agenerate(messages)

    @pytest.mark.asyncio
    async def test_citation_mixin_context_tag_processing(self):
        """Test that context tags are processed correctly."""

        model = MockChatModel(response_content="Response with context")

        # Create a message with context tags
        context_message = HumanMessage(
            content="Question about <context key='test'>\nSome context\n</context>"
        )
        messages: list[list[BaseMessage]] = [[context_message]]

        with patch(
            "langchain_b12.citations.citations.add_citations"
        ) as mock_add_citations:

            # Simulate structured content after citation processing
            cited_message = AIMessage(
                content=[
                    {
                        "text": "Response with context",
                        "citations": [
                            {
                                "cited_text": "Some context",
                                "generated_cited_text": "Some context",
                                "key": "test",
                                "dist": 0,
                            }
                        ],
                        "type": "text",
                    }
                ]
            )
            mock_add_citations.return_value = cited_message

            result = await model.agenerate(messages)

            # Should call add_citations with the context and response
            mock_add_citations.assert_called_once()

            # Content should be structured after processing
            generation = result.generations[0][0]
            assert generation.message == cited_message

    @pytest.mark.asyncio
    async def test_citation_mixin_context_tag_processing_invoke(self):
        """Test that context tags are processed correctly."""

        model = MockChatModel(response_content="Response with context")

        # Create a message with context tags
        context_message = HumanMessage(
            content="Question about <context key='test'>\nSome context\n</context>"
        )
        messages: list[BaseMessage] = [context_message]

        with patch(
            "langchain_b12.citations.citations.add_citations"
        ) as mock_add_citations:

            # Simulate structured content after citation processing
            cited_message = AIMessage(
                content=[
                    {
                        "text": "Response with context",
                        "citations": [
                            {
                                "cited_text": "Some context",
                                "generated_cited_text": "Some context",
                                "key": "test",
                                "dist": 0,
                            }
                        ],
                        "type": "text",
                    }
                ]
            )
            mock_add_citations.return_value = cited_message

            result = await model.ainvoke(messages)

            # Should call add_citations with the context and response
            mock_add_citations.assert_called_once()

            # Content should be structured after processing
            assert result == cited_message

    @pytest.mark.asyncio
    async def test_citation_mixin_kwargs_preservation(self):
        """Test that kwargs are properly passed through."""

        class KwargsTestModel(CitationMixin):
            @property
            def _llm_type(self) -> str:
                return "kwargstest"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                from langchain_core.outputs import ChatResult

                ai_message = AIMessage(content="Kwargs test")
                generation = ChatGeneration(message=ai_message)
                return ChatResult(generations=[generation])

        model = KwargsTestModel()
        messages: list[list[BaseMessage]] = [[HumanMessage(content="Test message")]]

        test_kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "custom_param": "test_value",
        }

        with patch.object(
            CitationMixin.__bases__[0], "agenerate"
        ) as mock_parent_agenerate:

            mock_parent_agenerate.return_value = LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="Kwargs test"))]]
            )

            await model.agenerate(messages, **test_kwargs)

            # Check that kwargs were passed to parent's agenerate
            mock_parent_agenerate.assert_called_once()
            _, call_kwargs = mock_parent_agenerate.call_args

            # Verify all test kwargs are present
            for key, value in test_kwargs.items():
                assert key in call_kwargs
                assert call_kwargs[key] == value

    @pytest.mark.asyncio
    async def test_citation_mixin_multiple_message_batches(self):
        """Test handling of multiple message batches."""

        model = MockChatModel()
        messages: list[list[BaseMessage]] = [
            [HumanMessage(content="First batch")],
            [HumanMessage(content="Second batch")],
        ]

        with patch.object(
            CitationMixin.__bases__[0], "agenerate"
        ) as mock_parent_agenerate:

            mock_parent_agenerate.return_value = LLMResult(
                generations=[
                    [ChatGeneration(message=AIMessage(content="First response"))],
                    [ChatGeneration(message=AIMessage(content="Second response"))],
                ]
            )

            result = await model.agenerate(messages)

            # Should process all batches
            assert len(result.generations) == 2
            assert len(result.generations[0]) == 1
            assert len(result.generations[1]) == 1

    @pytest.mark.asyncio
    async def test_citation_mixin_realistic_workflow(self):
        """Test a realistic workflow with context and citations."""

        model = MockChatModel()

        # Message with context tags
        messages: list[list[BaseMessage]] = [
            [
                HumanMessage(
                    content="What is the capital of France? "
                    "<context key='france'>\nFrance is a country in Europe. "
                    "Paris is the capital city.\n</context>"
                )
            ]
        ]

        with (
            patch.object(
                CitationMixin.__bases__[0], "agenerate"
            ) as mock_parent_agenerate,
            patch(
                "langchain_b12.citations.citations.add_citations"
            ) as mock_add_citations,
        ):

            mock_parent_agenerate.return_value = LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(content="The capital of France is Paris.")
                        )
                    ]
                ]
            )

            # Simulate citation processing result
            cited_message = AIMessage(
                content=[
                    {
                        "text": "The capital of France is Paris.",
                        "citations": [
                            {
                                "cited_text": "Paris is the capital city",
                                "generated_cited_text": "Paris is the capital city",
                                "key": "france",
                                "dist": 0,
                            }
                        ],
                        "type": "text",
                    }
                ]
            )
            mock_add_citations.return_value = cited_message

            result = await model.agenerate(messages)

            # Verify the workflow executed correctly
            mock_parent_agenerate.assert_called_once()
            mock_add_citations.assert_called_once()

            # Check that the result has the expected structure
            assert len(result.generations) == 1
            assert len(result.generations[0]) == 1

            # Verify structured content after citation processing
            generation = result.generations[0][0]
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.message.content, list)
            # Basic validation that citation processing occurred
            assert len(generation.message.content) > 0

    @pytest.mark.asyncio
    async def test_citation_mixin_tool_call_handling(self):
        """Test handling of AIMessage with tool calls (no string content)."""

        messages: list[list[BaseMessage]] = [[HumanMessage(content="Use a tool")]]
        tool_calls: list[ToolCall] = [
            {
                "name": "test_tool",
                "args": {"param": "value"},
                "id": "call_123",
                "type": "tool_call",
            }
        ]
        model = MockChatModel(response_content="", tool_calls=tool_calls)

        result = await model.agenerate(messages)

        # Should handle tool calls without issues
        assert len(result.generations) == 1
        generation = result.generations[0][0]
        assert isinstance(generation, ChatGeneration)
        assert isinstance(generation.message, AIMessage)
        # Content remains empty for tool calls
        assert generation.message.content == ""
        # Tool calls should be preserved
        assert generation.message.tool_calls == tool_calls

    @pytest.mark.asyncio
    async def test_citation_mixin_error_in_citation_processing(self):
        """Test that errors in citation processing are handled properly."""

        model = MockChatModel()
        messages: list[list[BaseMessage]] = [
            [
                HumanMessage(
                    content="Test question <context key='test'>Some context</context>"
                )
            ]
        ]

        with patch(
            "langchain_b12.citations.citations.add_citations"
        ) as mock_add_citations:

            # Simulate an error in citation processing
            mock_add_citations.side_effect = RuntimeError("Citation error")

            # Should propagate the error from citation processing
            with pytest.raises(RuntimeError, match="Citation error"):
                await model.agenerate(messages)
