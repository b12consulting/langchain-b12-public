import logging
import os
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from operator import itemgetter
from typing import Any, Literal, cast

from google import genai
from google.genai import types
from google.oauth2 import service_account
from langchain_b12.genai.genai_utils import (
    convert_messages_to_contents,
    parse_response_candidate,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LangSmithParams, LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ChatGenAI(BaseChatModel):
    """Implementation of BaseChatModel using `google-genai`"""

    client: genai.Client = Field(
        default_factory=lambda: genai.Client(
            vertexai=True,
            credentials=service_account.Credentials.from_service_account_file(
                filename=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            ),
        ),
        exclude=True,
    )
    model_name: str = Field(default="gemini-2.0-flash", alias="model")
    "Underlying model name."
    stop: list[str] | None = Field(default=None, alias="stop_sequences")
    "Optional list of stop words to use when generating."
    temperature: float | None = None
    "Sampling temperature, it controls the degree of randomness in token selection."
    max_output_tokens: int | None = Field(default=None, alias="max_tokens")
    "Token limit determines the maximum amount of text output from one prompt."
    top_p: float | None = None
    "Tokens are selected from most probable to least until the sum of their "
    "probabilities equals the top-p value. Top-p is ignored for Codey models."
    top_k: int | None = None
    "How the model selects tokens for output, the next token is selected from "
    "among the top-k most probable tokens. Top-k is ignored for Codey models."
    n: int = 1
    """How many completions to generate for each prompt."""
    seed: int | None = None
    """Random seed for the generation."""
    safety_settings: list[types.SafetySetting] | None = None
    """The default safety settings to use for all generations.

        For example:

            from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            """  # noqa: E501
    thinking_config: types.ThinkingConfig | None = None
    "The thinking configuration to use for the model."

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def _llm_type(self) -> str:
        return "vertexai"

    @property
    def _default_params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
            "seed": self.seed,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Gets the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_b12", "genai", "genai"]

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = {**self._default_params, **kwargs}
        ls_params = LangSmithParams(
            ls_provider="google_vertexai",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_output_tokens", self.max_output_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _prepare_request(
        self, messages: list[BaseMessage]
    ) -> tuple[str | None, types.ContentListUnion]:
        contents = convert_messages_to_contents(messages)
        system_message: SystemMessage | None = next(
            (message for message in messages if isinstance(message, SystemMessage)),
            None,
        )
        system_instruction = system_message.content if system_message else None
        assert system_instruction is None or isinstance(
            system_instruction, str
        ), "System message content must be a string or None"
        return system_instruction, cast(types.ContentListUnion, contents)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        contents = convert_messages_to_contents([HumanMessage(content=text)])
        response = self.client.models.count_tokens(
            model=self.model_name,
            contents=cast(types.ContentListUnion, contents),
        )
        return response.total_tokens or 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        stream_iter = self._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return generate_from_stream(stream_iter)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        stream_iter = self._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return await agenerate_from_stream(stream_iter)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        system_message, contents = self._prepare_request(messages=messages)
        response_iter = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_output_tokens=self.max_output_tokens,
                candidate_count=self.n,
                stop_sequences=stop or self.stop,
                safety_settings=self.safety_settings,
                thinking_config=self.thinking_config,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
                **kwargs,
            ),
        )
        total_lc_usage = None
        for response_chunk in response_iter:
            chunk, total_lc_usage = self._gemini_chunk_to_generation_chunk(
                response_chunk, prev_total_usage=total_lc_usage
            )
            if run_manager and isinstance(chunk.message.content, str):
                run_manager.on_llm_new_token(chunk.message.content)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        system_message, contents = self._prepare_request(messages=messages)
        response_iter = self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_output_tokens=self.max_output_tokens,
                candidate_count=self.n,
                stop_sequences=stop or self.stop,
                safety_settings=self.safety_settings,
                thinking_config=self.thinking_config,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
                **kwargs,
            ),
        )
        total_lc_usage = None
        async for response_chunk in await response_iter:
            chunk, total_lc_usage = self._gemini_chunk_to_generation_chunk(
                response_chunk, prev_total_usage=total_lc_usage
            )
            if run_manager and isinstance(chunk.message.content, str):
                await run_manager.on_llm_new_token(chunk.message.content)
            yield chunk

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        method: Literal["json_mode", "function_calling"] = "json_mode",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        assert isinstance(schema, type) and issubclass(
            schema, BaseModel
        ), "Structured output is only supported for Pydantic models."
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        parser: OutputParserLike

        if method == "json_mode":
            parser = PydanticOutputParser(pydantic_object=schema)

            llm = self.bind(
                response_mime_type="application/json",
                response_schema=schema,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
        elif method == "function_calling":
            parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            tool_choice = schema.__name__
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_choice,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": convert_to_openai_tool(schema),
                },
            )
        else:
            raise ValueError("method must be either 'json_mode' or 'function_calling'.")

        if include_raw:
            parser_with_fallback = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
            ).with_fallbacks(
                [RunnablePassthrough.assign(parsed=lambda _: None)],
                exception_key="parsing_error",
            )
            return {"raw": llm} | parser_with_fallback
        else:
            return llm | parser

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type | Callable[..., Any] | BaseTool | types.Tool
        ],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with Vertex tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = []
        for tool in tools:
            if not isinstance(tool, types.Tool):
                openai_tool = convert_to_openai_tool(tool)
                if openai_tool["type"] != "function":
                    raise ValueError(
                        f"Tool {tool} is not a function tool. "
                        f"It is a {openai_tool['type']}. "
                        "Only function tools are supported."
                    )
                function = openai_tool["function"]
                formatted_tools.append(
                    types.Tool(
                        function_declarations=[types.FunctionDeclaration(**function)],
                    )
                )
            else:
                formatted_tools.append(tool)
        if tool_choice:
            kwargs["tool_config"] = types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=[tool_choice],
            )
        return self.bind(tools=formatted_tools, **kwargs)

    def _gemini_chunk_to_generation_chunk(
        self,
        response_chunk: types.GenerateContentResponse,
        prev_total_usage: UsageMetadata | None = None,
    ) -> tuple[ChatGenerationChunk, UsageMetadata | None]:
        def _parse_usage_metadata(
            usage_metadata: types.GenerateContentResponseUsageMetadata,
        ) -> UsageMetadata:
            return UsageMetadata(
                input_tokens=usage_metadata.prompt_token_count or 0,
                output_tokens=usage_metadata.candidates_token_count or 0,
                total_tokens=usage_metadata.total_token_count or 0,
                input_token_details={
                    "cache_read": usage_metadata.cached_content_token_count or 0
                },
                output_token_details={
                    "reasoning": usage_metadata.thoughts_token_count or 0
                },
            )

        total_lc_usage: UsageMetadata | None = (
            _parse_usage_metadata(response_chunk.usage_metadata)
            if response_chunk.usage_metadata
            else None
        )

        if total_lc_usage and prev_total_usage:
            lc_usage: UsageMetadata | None = UsageMetadata(
                input_tokens=total_lc_usage["input_tokens"]
                - prev_total_usage["input_tokens"],
                output_tokens=total_lc_usage["output_tokens"]
                - prev_total_usage["output_tokens"],
                total_tokens=total_lc_usage["total_tokens"]
                - prev_total_usage["total_tokens"],
            )
        else:
            lc_usage = total_lc_usage
        if not response_chunk.candidates:
            message = AIMessageChunk(content="")
            if lc_usage:
                message.usage_metadata = lc_usage
            generation_info = {}
        else:
            top_candidate = response_chunk.candidates[0]
            generation_info = {
                "finish_reason": top_candidate.finish_reason,
                "finish_message": top_candidate.finish_message,
            }
            try:
                message = parse_response_candidate(top_candidate)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse model response: {top_candidate.finish_message}"
                ) from e
            if lc_usage:
                message.usage_metadata = lc_usage
            # add model name if final chunk
            if top_candidate.finish_reason is not None:
                message.response_metadata["model_name"] = self.model_name

        return (
            ChatGenerationChunk(
                message=message,
                generation_info=generation_info,
            ),
            total_lc_usage,
        )
