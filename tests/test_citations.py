from unittest.mock import AsyncMock

import pytest
from langchain_b12.citations.citations import (
    SYSTEM_PROMPT,
    Citation,
    Citations,
    Match,
    add_citations,
    contains_context_tags,
    create_citation_model,
    merge_citations,
    split_into_sentences,
    validate_citations,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class TestUtilityFunctions:
    """Test utility functions for citation processing."""

    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        # Basic sentences
        text = "This is sentence one. This is sentence two! This is sentence three?"
        sentences = split_into_sentences(text)
        assert sentences == [
            "This is sentence one.",
            " This is sentence two!",
            " This is sentence three?",
        ]

        # Single sentence
        text = "Just one sentence."
        sentences = split_into_sentences(text)
        assert sentences == ["Just one sentence."]

        # Empty text
        text = ""
        sentences = split_into_sentences(text)
        assert sentences == [""]

        # Text with multiple spaces
        text = "First sentence.    Second sentence!"
        sentences = split_into_sentences(text)
        assert sentences == ["First sentence.", "    Second sentence!"]

        # Text with no punctuation
        text = "No punctuation here"
        sentences = split_into_sentences(text)
        assert sentences == ["No punctuation here"]

    def test_split_into_sentences_preserves_text(self):
        """Test that joining split sentences recreates the original text."""
        test_cases = [
            "This is sentence one. This is sentence two! This is sentence three?",
            "Just one sentence.",
            "",
            "First sentence.    Second sentence!",
            "No punctuation here",
            "What?! Really... Yes.",
            "Dr. Smith went home. He was tired.",
            "Multiple   spaces.     Between sentences!",
            "Text with newlines.\nSecond line. Third sentence!",
        ]

        for text in test_cases:
            sentences = split_into_sentences(text)
            reconstructed = "".join(sentences)
            assert reconstructed == text, f"Failed for text: {repr(text)}"

    def test_contains_context_tags(self):
        """Test context tag detection."""
        # Text with context tags
        text = 'Some text <context key="test">Context content</context> more text'
        assert contains_context_tags(text) is True

        # Text without context tags
        text = "Just regular text without any tags"
        assert contains_context_tags(text) is False

        # Multiple context tags
        text = """
        <context key="one">First context</context>
        Some text here
        <context key="two">Second context</context>
        """
        assert contains_context_tags(text) is True

        # Multiline context
        text = """
        <context key="multiline">
        This is a multiline
        context with multiple
        lines of content
        </context>
        """
        assert contains_context_tags(text) is True

        # Empty string
        assert contains_context_tags("") is False

    def test_merge_citations(self):
        """Test merging citations with sentences."""
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        citations_with_matches: list[tuple[Citation, Match | None]] = [
            (
                Citation(sentence_index=0, cited_text="source text 1", key="key1"),
                {"start": 0, "end": 13, "dist": 0, "matched": "source text 1"},
            ),
            (
                Citation(sentence_index=0, cited_text="source text 2", key="key2"),
                {"start": 0, "end": 13, "dist": 1, "matched": "source text 2"},
            ),
            (
                Citation(sentence_index=2, cited_text="source text 3", key="key3"),
                {"start": 0, "end": 13, "dist": 2, "matched": "source text 3"},
            ),
            (
                Citation(sentence_index=2, cited_text="source text 4", key="key4"),
                None,
            ),
        ]

        result = merge_citations(sentences, citations_with_matches)

        assert len(result) == 3

        # First sentence has two citations
        assert result[0]["text"] == "First sentence."
        assert result[0]["type"] == "text"
        assert len(result[0]["citations"]) == 2
        assert result[0]["citations"][0]["cited_text"] == "source text 1"
        assert result[0]["citations"][0]["key"] == "key1"
        assert result[0]["citations"][0]["dist"] == 0
        assert result[0]["citations"][1]["cited_text"] == "source text 2"
        assert result[0]["citations"][1]["key"] == "key2"
        assert result[0]["citations"][1]["dist"] == 1

        # Second sentence has no citations
        assert result[1]["text"] == "Second sentence."
        assert result[1]["citations"] is None

        # Third sentence has two citations
        assert result[2]["text"] == "Third sentence."
        assert len(result[2]["citations"]) == 2
        assert result[2]["citations"][0]["cited_text"] == "source text 3"
        assert result[2]["citations"][0]["key"] == "key3"
        assert result[2]["citations"][0]["dist"] == 2

        assert result[2]["citations"][1]["cited_text"] is None
        assert result[2]["citations"][1]["key"] == "key4"
        assert result[2]["citations"][1]["dist"] is None
        assert result[2]["citations"][1]["generated_cited_text"] == "source text 4"

    def test_validate_citations(self):
        """Test citation validation."""
        sentences = ["First sentence.", "Second sentence."]
        messages = [
            HumanMessage(
                content="<context key='test'>This is valid cited text</context>"
            ),
            HumanMessage(content="Some other content with invalid text"),
        ]

        citations = Citations(
            values=[
                Citation(
                    sentence_index=0, cited_text="This is valid cited text", key="test"
                ),
                Citation(
                    sentence_index=1,
                    cited_text="This text is not in messages",
                    key="test",
                ),
                Citation(
                    sentence_index=5, cited_text="This is valid cited text", key="test"
                ),  # Invalid index
                Citation(
                    sentence_index=-1, cited_text="This is valid cited text", key="test"
                ),  # Invalid index
            ]
        )

        validated = validate_citations(citations, messages, sentences)

        # Should return list of tuples with citations and matches
        assert isinstance(validated, list)
        assert len(validated) == 2  # Two valid citations (indexes 0 and 1)

        # Each item should be a tuple of (Citation, Match | None)
        citation, match = validated[0]
        assert isinstance(citation, Citation)
        assert citation.sentence_index == 0
        assert citation.cited_text == "This is valid cited text"
        assert citation.key == "test"
        assert match is not None
        assert match["dist"] >= 0  # Should have some fuzzy match distance
        assert match["matched"] == "This is valid cited text"

        # Second citation should also be included, fuzzy matching allows partial matches
        citation2, match2 = validated[1]
        assert citation2.sentence_index == 1
        assert citation2.cited_text == "This text is not in messages"
        assert citation2.key == "test"
        assert match2 is None


class TestAddCitations:
    """Test the main add_citations function."""

    @pytest.mark.asyncio
    async def test_add_citations_empty_content(self):
        """Test add_citations with empty message content."""
        model = AsyncMock(spec=BaseChatModel)
        messages = [HumanMessage(content="test")]
        message = AIMessage(content="")

        result = await add_citations(model, messages, message, SYSTEM_PROMPT)

        assert result == message
        model.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_citations_non_string_content(self):
        """Test add_citations with non-string content."""
        model = AsyncMock(spec=BaseChatModel)
        messages = [HumanMessage(content="test")]
        message = AIMessage(content=["not", "a", "string"])

        with pytest.raises(
            AssertionError,
            match="Citation agent currently only supports string content",
        ):
            await add_citations(model, messages, message, SYSTEM_PROMPT)

    @pytest.mark.asyncio
    async def test_add_citations_no_context_tags(self):
        """Test add_citations with content that has no context tags."""
        model = AsyncMock(spec=BaseChatModel)
        messages = [HumanMessage(content="test")]
        message = AIMessage(content="This is a message without context tags.")

        result = await add_citations(model, messages, message, SYSTEM_PROMPT)

        assert result == message
        model.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_citations_with_context_tags(self):
        """Test add_citations with content that has context tags."""
        # Mock the model
        model = AsyncMock(spec=BaseChatModel)
        structured_output_mock = AsyncMock()
        model.with_structured_output.return_value = structured_output_mock

        # Mock citations response
        mock_citations = Citations(
            values=[Citation(sentence_index=0, cited_text="test context", key="key1")]
        )
        structured_output_mock.ainvoke.return_value = mock_citations

        # Setup messages and content - the AI message needs to contain context tags
        messages = [
            HumanMessage(content='<context key="key1">test context</context>'),
        ]
        message = AIMessage(
            content='This sentence uses <context key="key1">test context</context>.'
        )

        result = await add_citations(model, messages, message, SYSTEM_PROMPT)

        # Verify model was called
        model.with_structured_output.assert_called_once_with(Citations)
        structured_output_mock.ainvoke.assert_called_once()

        # Check the call arguments
        call_args = structured_output_mock.ainvoke.call_args[0][0]
        assert (
            len(call_args) == 3
        )  # system message + original message + numbered message
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == SYSTEM_PROMPT
        assert isinstance(call_args[2], AIMessage)  # numbered message
        assert "0: This sentence uses" in call_args[2].content

        # Verify result structure
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert "This sentence uses" in result.content[0]["text"]
        assert result.content[0]["type"] == "text"
        assert len(result.content[0]["citations"]) == 1
        # The cited_text might be modified by fuzzy matching
        assert result.content[0]["citations"][0]["key"] == "key1"
        assert "dist" in result.content[0]["citations"][0]

    @pytest.mark.asyncio
    async def test_add_citations_multiple_sentences(self):
        """Test add_citations with multiple sentences."""
        model = AsyncMock(spec=BaseChatModel)
        structured_output_mock = AsyncMock()
        model.with_structured_output.return_value = structured_output_mock

        mock_citations = Citations(
            values=[
                Citation(sentence_index=0, cited_text="first context", key="key1"),
                Citation(sentence_index=1, cited_text="second context", key="key2"),
            ]
        )
        structured_output_mock.ainvoke.return_value = mock_citations

        messages = [
            HumanMessage(content='<context key="key1">first context</context>'),
            HumanMessage(content='<context key="key2">second context</context>'),
        ]
        message = AIMessage(content="First sentence. Second sentence!")

        result = await add_citations(model, messages, message, SYSTEM_PROMPT)

        # Check numbered message formatting
        call_args = structured_output_mock.ainvoke.call_args[0][0]
        numbered_content = call_args[-1].content
        assert "0: First sentence." in numbered_content
        assert "1: Second sentence!" in numbered_content

        # Verify result
        assert len(result.content) == 2
        assert len(result.content[0]["citations"]) == 1
        assert len(result.content[1]["citations"]) == 1
        assert result.content[0]["citations"][0]["key"] == "key1"
        assert result.content[1]["citations"][0]["key"] == "key2"


class TestCreateCitationModel:
    """Test the create_citation_model function."""

    @pytest.mark.asyncio
    async def test_create_citation_model_basic(self):
        """Test basic citation model creation."""
        base_model = AsyncMock(spec=BaseChatModel)
        base_model.ainvoke.return_value = AIMessage(
            content="Response without context tags."
        )

        citation_model = create_citation_model(base_model)

        messages = [HumanMessage(content="Test message")]
        result = await citation_model.ainvoke(messages)

        base_model.ainvoke.assert_called_once_with(messages)
        assert isinstance(result, AIMessage)
        assert result.content == "Response without context tags."

    @pytest.mark.asyncio
    async def test_create_citation_model_end_to_end(self):
        """Test complete end-to-end citation model functionality."""
        base_model = AsyncMock(spec=BaseChatModel)
        citation_model_instance = AsyncMock(spec=BaseChatModel)

        # Setup base model response with context tags
        base_model.ainvoke.return_value = AIMessage(
            content="The sky is blue. Grass is green."
        )

        # Setup citation model response
        structured_output_mock = AsyncMock()
        citation_model_instance.with_structured_output.return_value = (
            structured_output_mock
        )
        mock_citations = Citations(
            values=[
                Citation(sentence_index=0, cited_text="sky is blue", key="weather"),
                Citation(sentence_index=1, cited_text="grass is green", key="nature"),
            ]
        )
        structured_output_mock.ainvoke.return_value = mock_citations

        citation_model = create_citation_model(
            base_model, citation_model=citation_model_instance
        )

        messages = [
            HumanMessage(
                content='<context key="weather">The sky is blue today</context>'
            ),
            HumanMessage(
                content='<context key="nature">The grass is green in spring</context>'
            ),
        ]

        result = await citation_model.ainvoke(messages)

        # Verify the complete flow
        base_model.ainvoke.assert_called_once_with(messages)
        citation_model_instance.with_structured_output.assert_called_once_with(
            Citations
        )
        structured_output_mock.ainvoke.assert_called_once()

        # Check final result structure
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)
        assert len(result.content) == 2

        # First sentence
        assert result.content[0]["text"] == "The sky is blue."
        # The cited_text might be modified by fuzzy matching
        assert result.content[0]["citations"][0]["key"] == "weather"
        assert "dist" in result.content[0]["citations"][0]

        # Second sentence
        assert result.content[1]["text"] == " Grass is green."
        # The cited_text might be modified by fuzzy matching
        assert result.content[1]["citations"][0]["key"] == "nature"
        assert "dist" in result.content[1]["citations"][0]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_split_into_sentences_edge_cases(self):
        """Test edge cases for sentence splitting."""
        # Sentences ending with multiple punctuation
        text = "What?! Really... Yes."
        sentences = split_into_sentences(text)
        # The regex splits on any punctuation followed by space, so this creates 3 sentences
        assert len(sentences) == 3
        assert sentences == ["What?!", " Really...", " Yes."]

        # Abbreviations (potential issue with splitting)
        text = "Dr. Smith went home. He was tired."
        sentences = split_into_sentences(text)
        # This will incorrectly split at "Dr. " - this is a known limitation
        assert len(sentences) == 3  # "Dr.", " Smith went home.", " He was tired."

    def test_contains_context_tags_malformed(self):
        """Test context tag detection with malformed tags."""
        # Incomplete opening tag
        text = "<context key='test'>content</context>"
        assert contains_context_tags(text) is True

        # Missing key attribute
        text = "<context>content</context>"
        assert contains_context_tags(text) is False

        # Nested tags (shouldn't match)
        text = "<outer><context key='test'>content</context></outer>"
        assert contains_context_tags(text) is True

    def test_validate_citations_empty_inputs(self):
        """Test citation validation with empty inputs."""
        # Empty citations
        citations = Citations(values=[])
        messages = [HumanMessage(content="test")]
        sentences = ["test"]

        result = validate_citations(citations, messages, sentences)
        assert len(result) == 0

        # Empty messages - citations should still be returned with match or None
        citations = Citations(
            values=[Citation(sentence_index=0, cited_text="test", key="key")]
        )
        messages = []
        sentences = ["test"]

        result = validate_citations(citations, messages, sentences)
        assert len(result) == 1
        citation, match = result[0]
        assert citation.cited_text == "test"
        assert citation.key == "key"
        assert match is None

        # Empty sentences
        citations = Citations(
            values=[Citation(sentence_index=0, cited_text="test", key="key")]
        )
        messages = [HumanMessage(content="test")]
        sentences = []

        result = validate_citations(citations, messages, sentences)
        assert len(result) == 0
