import re
from collections.abc import Sequence
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
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


class CitationType(TypedDict):

    cited_text: str
    key: str


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
    """Split text into sentences on punctuation marks."""
    return re.split(r"(?<=[.!?]) +", text.strip())


def contains_context_tags(text: str) -> bool:
    """Check if the text contains context tags."""
    return bool(re.search(r"<context\s+key=[^>]+>.*?</context>", text, re.DOTALL))


def merge_citations(sentences: list[str], citations: Citations) -> list[ContentType]:
    """Merge citations into sentences."""
    content: list[ContentType] = []
    for sentence_index, sentence in enumerate(sentences):
        _citations: list[CitationType] = []
        for citation in citations.values:
            if citation.sentence_index == sentence_index:
                _citations.append(
                    {"cited_text": citation.cited_text, "key": citation.key}
                )
        content.append(
            {"text": sentence, "citations": _citations or None, "type": "text"}
        )

    return content


def validate_citations(
    citations: Citations,
    messages: Sequence[BaseMessage],
    sentences: list[str],
) -> Citations:
    """Validate the citations. Invalid citations are dropped."""
    n_sentences = len(sentences)

    all_text = "\n".join(
        str(msg.content) for msg in messages if isinstance(msg.content, str)
    )

    _citations: list[Citation] = []
    for citation in citations.values:
        if citation.sentence_index < 0 or citation.sentence_index >= n_sentences:
            continue
        if citation.cited_text not in all_text:
            continue
        _citations.append(citation)
    return Citations(values=_citations)


async def add_citations(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    message: AIMessage,
    system_prompt: str,
) -> AIMessage:
    """Add citations to the message."""
    if not message.content:
        # Nothing to be done, for example in case of a tool call
        return message

    assert isinstance(
        message.content, str
    ), "Citation agent currently only supports string content."

    if not contains_context_tags(message.content):
        # No context tags, nothing to do
        return message

    sentences = split_into_sentences(message.content)

    num_width = len(str(len(sentences)))
    numbered_message = AIMessage(
        content="\n".join(
            f"{str(i).rjust(num_width)}: {sentence}"
            for i, sentence in enumerate(sentences)
        ),
        name=message.name,
    )
    system_message = SystemMessage(system_prompt)
    _messages = [system_message, *messages, numbered_message]

    citations = await model.with_structured_output(Citations).ainvoke(_messages)
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
        content= {
            "citations": [
                {
                    "cited_text": "Today is a sunny day",
                    "key": "abc"
                }
            ],
            "text": "The grass is green",
            "type": "text",
        },
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
