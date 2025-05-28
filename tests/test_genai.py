from unittest.mock import MagicMock

from google.genai import Client, types
from langchain_b12.genai.genai import ChatGenAI
from langchain_core.messages import HumanMessage


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
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text="bar")])
                    ),
                ]
            ),
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text="baz")])
                    ),
                ]
            ),
        )
    )
    model = ChatGenAI(client=client)
    messages = [HumanMessage(content="foo")]
    response = model.invoke(messages)
    method: MagicMock = client.models.generate_content_stream
    method.assert_called_once()
    assert response.content == "barbaz"
