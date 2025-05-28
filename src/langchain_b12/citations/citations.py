from collections.abc import Sequence
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable


class Citation(TypedDict):
    """A class representing a citation."""

    cited_text: str
    document_index: int
    document_title: str
    end_char_index: int
    start_char_index: int
    type: Literal["char_location"]


def add_citation(
    message: BaseMessage,
) -> AIMessage:
    """Add a citation to the message."""
    pass


def create_citation_model(
    model: BaseChatModel,
) -> Runnable[Sequence[BaseMessage], AIMessage]:
    """Take a base chat model and wrap it such that it adds citations to the messages.
    The AIMessage will have the following structure:
    AIMessage(
        content= {
            "citations": [
                {
                    "cited_text": "The grass is green. ",
                    "document_index": 0,
                    "document_title": "My Document",
                    "end_char_index": 20,
                    "start_char_index": 0,
                    "type": "char_location",
                }
            ],
            "text": "the grass is green",
            "type": "text",
        },
    )
    """
    return model | add_citation
