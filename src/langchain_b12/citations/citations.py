import re
from collections.abc import Sequence
from typing import Any, Literal, TypedDict
from uuid import UUID

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from langchain_core.runnables import Runnable
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are an expert at identifying and adding citations to text.
Your task is to identify, for each sentence in the final message, which citations were used to generate it.

You will receive a numbered zero-indexed list of sentences in the final message, e.g.
```
0: Grass is green.
1: The sky is blue and the sun is shining.
```
The rest of the conversation may contain contexts enclosed in xml tags, e.g.
```
<context key="abc">
Today is a sunny day and the color of the grass is green.
</context>
```
Each sentence may have zero, one, or multiple citations from the contexts.
Each citation may be used for zero, one or multiple sentences.
A context may be cited zero, one, or multiple times.

The final message will be based on the contexts, but may not mention them explicitly.
You must identify which contexts and which parts of the contexts were used to generate each sentence.
For each such case, you must return a citation with a "sentence_index", "cited_text" and "key" property.
The "sentence_index" is the index of the sentence in the final message.
The "cited_text" must be a substring of the full context that was used to generate the sentence.
The "key" must be the key of the context that was used to generate the sentence.
Make sure that you copy the "cited_text" verbatim from the context, or it will not be considered valid.

For the example above, the output should look like this:
[
    {
        "sentence_index": 0,
        "cited_text": "the color of the grass is green",
        "key": "abc"
    },
    {
        "sentence_index": 1,
        "cited_text": "Today is a sunny day",
        "key": "abc"
    },
]
""".strip()  # noqa: E501


class Match(TypedDict):
    start: int
    end: int
    dist: int
    matched: str


class CitationType(TypedDict):

    cited_text: str | None
    generated_cited_text: str
    key: str
    dist: int | None


class ContentType(TypedDict):

    citations: list[CitationType] | None
    text: str
    type: Literal["text"]


class Citation(BaseModel):

    sentence_index: int = Field(
        ...,
        description="The index of the sentence from your answer "
        "that this citation refers to.",
    )
    cited_text: str = Field(
        ...,
        description="The text that is cited from the document. "
        "Make sure you cite it verbatim!",
    )
    key: str = Field(..., description="The key of the document you are citing.")


class Citations(BaseModel):

    values: list[Citation] = Field(..., description="List of citations")


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation marks and newlines."""
    if not text:
        return [text]

    # Split after punctuation followed by spaces, or on newlines
    # Use capturing groups to preserve delimiters (spaces and newlines)
    parts = re.split(r"((?<=[.!?])(?= +)|\n+)", text)

    # Filter out empty strings that can result from splitting
    return [part for part in parts if part]


def contains_context_tags(text: str) -> bool:
    """Check if the text contains context tags."""
    return bool(re.search(r"<context\s+key=[^>]+>.*?</context>", text, re.DOTALL))


def merge_citations(
    sentences: list[str], citations: list[tuple[Citation, Match | None]]
) -> list[ContentType]:
    """Merge citations into sentences."""
    content: list[ContentType] = []
    for sentence_index, sentence in enumerate(sentences):
        _citations: list[CitationType] = []
        for citation, match in citations:
            if citation.sentence_index == sentence_index:
                if match is None:
                    _citations.append(
                        {
                            "cited_text": None,
                            "generated_cited_text": citation.cited_text,
                            "key": citation.key,
                            "dist": None,
                        }
                    )
                else:
                    _citations.append(
                        {
                            "cited_text": match["matched"],
                            "generated_cited_text": citation.cited_text,
                            "key": citation.key,
                            "dist": match["dist"],
                        }
                    )
        content.append(
            {"text": sentence, "citations": _citations or None, "type": "text"}
        )

    return content


def validate_citations(
    citations: Citations,
    messages: Sequence[BaseMessage],
    sentences: list[str],
) -> list[tuple[Citation, Match | None]]:
    """Validate the citations. Invalid citations are dropped."""
    from fuzzysearch import find_near_matches

    n_sentences = len(sentences)

    all_text = "\n".join(
        str(msg.content) for msg in messages if isinstance(msg.content, str)
    )

    citations_with_matches: list[tuple[Citation, Match | None]] = []
    for citation in citations.values:
        if citation.sentence_index < 0 or citation.sentence_index >= n_sentences:
            # discard citations that refer to non-existing sentences
            continue
        # Allow for 10% error distance
        max_l_dist = max(1, len(citation.cited_text) // 10)
        matches = find_near_matches(
            citation.cited_text, all_text, max_l_dist=max_l_dist
        )
        if not matches:
            citations_with_matches.append((citation, None))
        else:
            match = matches[0]
            citations_with_matches.append(
                (
                    citation,
                    Match(
                        start=match.start,
                        end=match.end,
                        dist=match.dist,
                        matched=match.matched,
                    ),
                )
            )
    return citations_with_matches


async def add_citations(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    message: AIMessage,
    system_prompt: str,
    **kwargs: Any,
) -> AIMessage:
    """Add citations to the message."""
    if not message.content:
        # Nothing to be done, for example in case of a tool call
        return message

    assert isinstance(
        message.content, str
    ), "Citation agent currently only supports string content."

    if not contains_context_tags("\n".join(str(msg.content) for msg in messages)):
        # No context tags, nothing to do
        return message

    sentences = split_into_sentences(message.content)

    num_width = len(str(len(sentences)))
    numbered_message = AIMessage(
        content="\n".join(
            f"{str(i).rjust(num_width)}: {sentence.strip()}"
            for i, sentence in enumerate(sentences)
        ),
        name=message.name,
    )
    system_message = SystemMessage(system_prompt)
    _messages = [system_message, *messages, numbered_message]

    citations = await model.with_structured_output(Citations).ainvoke(
        _messages, **kwargs
    )
    assert isinstance(
        citations, Citations
    ), f"Expected Citations from model invocation but got {type(citations)}"
    citations = validate_citations(citations, messages, sentences)

    message.content = merge_citations(sentences, citations)  # type: ignore[assignment]
    return message


def create_citation_model(
    model: BaseChatModel,
    citation_model: BaseChatModel | None = None,
    system_prompt: str | None = None,
) -> Runnable[Sequence[BaseMessage], AIMessage]:
    """Take a base chat model and wrap it such that it adds citations to the messages.
    Any contexts to be cited should be provided in the messages as XML tags,
    e.g. `<context key="abc">Today is a sunny day</context>`.
    The returned AIMessage will have the following structure:
    AIMessage(
        content=[
            {
                "citations": [
                    {
                        "cited_text": "the color of the grass is green",
                        "generated_cited_text": "the color of the grass is green",
                        "key": "abc",
                        "dist": 0,
                    }
                ],
                "text": "The grass is green",
                "type": "text",
            },
            {
                "citations": None,
                "text": "Is there anything else I can help you with?",
                "type": "text",
            }
        ]
    )

    Args:
        model: The base chat model to wrap.
        citation_model: The model to use for extracting citations.
            If None, the base model is used.
        system_prompt: The system prompt to use for the citation model.
            If None, a default prompt is used.
    """
    citation_model = citation_model or model
    system_prompt = system_prompt or SYSTEM_PROMPT

    async def ainvoke_with_citations(
        messages: Sequence[BaseMessage],
    ) -> AIMessage:
        """Invoke the model and add citations to the AIMessage."""
        ai_message = await model.ainvoke(messages)
        assert isinstance(
            ai_message, AIMessage
        ), f"Expected AIMessage from model invocation but got {type(ai_message)}"
        return await add_citations(citation_model, messages, ai_message, system_prompt)

    return RunnableCallable(
        func=None,  # TODO: Implement a sync version if needed
        afunc=ainvoke_with_citations,
    )


class CitationMixin(BaseChatModel):
    """Mixin class to add citation functionality to a runnable.

    Example usage:
    ```
    from langchain_b12.genai.genai import ChatGenAI
    from langchain_b12.citations.citations import CitationMixin

    class CitationModel(ChatGenAI, CitationMixin):
        pass
    ```
    """

    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        # Check if we should generate citations and remove it from kwargs
        generate_citations = kwargs.pop("generate_citations", True)

        llm_result = await super().agenerate(
            messages,
            stop,
            callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            run_id=run_id,
            **kwargs,
        )

        # Prevent recursion when extracting citations
        if not generate_citations:
            # Below we are call `add_citations` which will call `agenerate` again
            # This will lead to an infinite loop if we don't stop here.
            # We explicitly pass `generate_citations=False` below to sto this recursion.
            return llm_result

        # overwrite each generation with a version that has citations added
        for _messages, generations in zip(messages, llm_result.generations):
            for generation in generations:
                assert isinstance(generation, ChatGeneration) and not isinstance(
                    generation, ChatGenerationChunk
                ), f"Expected ChatGeneration; received {type(generation)}"
                assert isinstance(
                    generation.message, AIMessage
                ), f"Expected AIMessage; received {type(generation.message)}"
                message_with_citations = await add_citations(
                    self,
                    _messages,
                    generation.message,
                    SYSTEM_PROMPT,
                    generate_citations=False,
                )
                generation.message = message_with_citations

        return llm_result
