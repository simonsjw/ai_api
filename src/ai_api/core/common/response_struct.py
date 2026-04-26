"""
Generic structured (JSON) response support for any LLM provider.

This module provides a thin, provider-agnostic layer for requesting
Pydantic-validated structured JSON output from chat completions. It keeps
the public API simple while allowing all provider-specific formatting and
parsing logic to live in the per-provider chat modules.

What it does
------------
- `create_json_response_spec` returns the correct provider-specific
  response-spec object (`OllamaJSONResponseSpec` or `xAIJSONResponseSpec`)
  given a Pydantic model or raw JSON schema.
- `generate_structured_json` and `generate_structured_json_stream` are
  convenience entry points that simply forward to `client.create_chat(...)`
  with the `response_model` argument. They exist so that users who prefer
  a functional style do not have to remember the exact client method.

How it does it
--------------
- The heavy lifting (schema conversion, attaching the spec to the request,
  parsing the final response into a Pydantic instance) is performed inside
  the provider-specific `chat_turn_*.py` and `chat_stream_*.py` files.
- This module only performs a small dispatch on the `provider` string and
  re-exports the two generic generator functions.
- Because modern providers (xAI and Ollama) natively support JSON schema
  enforcement, the optional `instruction` parameter is rarely needed and
  is kept only for advanced / legacy use cases.

Examples — usage by clients / end users
---------------------------------------
Preferred way (most clients expose it directly on `create_chat`):

.. code-block:: python

    from pydantic import BaseModel
    from ai_api.core.ollama_client import TurnOllamaClient
    import logging


    class Person(BaseModel):
        name: str
        age: int
        hobbies: list[str]


    logger = logging.getLogger(__name__)
    client = TurnOllamaClient(logger=logger)

    response = await client.create_chat(
        messages=[
            {
                "role": "user",
                "content": "Tell me about Alice, 28, who loves hiking and coding.",
            }
        ],
        model="llama3.2",
        response_model=Person,
    )
    print(response.parsed)  # -> Person(name='Alice', age=28, hobbies=['hiking', 'coding'])

Using the generic helpers (functional style):

.. code-block:: python

    from ai_api.core.common.response_struct import (
        create_json_response_spec,
        generate_structured_json,
    )

    spec = create_json_response_spec("ollama", Person)
    # spec is an OllamaJSONResponseSpec instance ready to attach to a request

    response = await generate_structured_json(client, messages, Person, model="llama3.2")

See Also
--------
ai_api.core.ollama.chat_turn_ollama, ai_api.core.xai.chat_turn_xai
    Where the actual schema attachment and response parsing occurs.
ai_api.data_structures.ollama_objects.OllamaJSONResponseSpec
ai_api.data_structures.xai_objects.xAIJSONResponseSpec
    The concrete spec classes returned by `create_json_response_spec`.
"""

from typing import Any, AsyncIterator, Type

from pydantic import BaseModel

from ...data_structures.base_objects import (
    LLMResponseProtocol,
    LLMStreamingChunkProtocol,
)

__all__: list[str] = [
    "create_json_response_spec",
    "generate_structured_json",
    "generate_structured_json_stream",
]


def create_json_response_spec(
    provider: str,
    model: type[BaseModel] | dict[str, Any],
    instruction: str | None = None,                                                       # ← Optional, no automatic default
) -> Any:
    """
    Return the appropriate JSON response spec for the given provider.

    The `instruction` parameter is kept for advanced/power-user cases but
    is completely optional. Modern providers (xAI + Ollama) enforce schemas
    natively, so extra instructions are rarely needed.

    Parameters
    ----------
    provider : {"xai", "ollama"}
        Target provider. Determines which spec class is instantiated.
    model : type[pydantic.BaseModel] or dict[str, Any]
        Either a Pydantic model class or a raw JSON schema dictionary.
    instruction : str or None, optional
        Optional extra guidance string (rarely required in 2026).

    Returns
    -------
    OllamaJSONResponseSpec or xAIJSONResponseSpec
        Provider-specific spec object ready to be attached to a request.

    Raises
    ------
    NotImplementedError
        If `provider` is not one of the supported values.
    """
    if provider == "xai":
        from ...data_structures.xai_objects import xAIJSONResponseSpec

        return xAIJSONResponseSpec(model=model, instruction=instruction)
    elif provider == "ollama":
        from ...data_structures.ollama_objects import OllamaJSONResponseSpec

        return OllamaJSONResponseSpec(model=model, instruction=instruction)
    else:
        raise NotImplementedError(
            f"Structured output not implemented for provider '{provider}'"
        )


async def generate_structured_json(
    client: Any,
    messages: list[dict] | str,
    response_model: type[BaseModel],
    model: str = "default",
    **kwargs: Any,
) -> LLMResponseProtocol:
    """
    Generic non-streaming structured output helper.

    Most users should simply call ``await client.create_chat(..., response_model=MyModel)``
    directly. This function exists for symmetry with the streaming variant and
    for code that prefers a functional style.

    Parameters
    ----------
    client : Any
        An object satisfying ``LLMProviderAdapter`` (i.e. has a ``create_chat`` method).
    messages : list[dict] or str
        Chat messages or a single prompt string.
    response_model : type[pydantic.BaseModel]
        Pydantic model that the final response must validate against.
    model : str, optional
        Model name (default "default" — provider-specific).
    **kwargs
        Forwarded verbatim to ``client.create_chat``.

    Returns
    -------
    LLMResponseProtocol
        The completed response (the concrete type depends on the client).
    """
    return await client.create_chat(
        messages=messages,
        model=model,
        response_model=response_model,
        **kwargs,
    )


async def generate_structured_json_stream(
    client: Any,
    messages: list[dict] | str,
    response_model: type[BaseModel],
    model: str = "default",
    **kwargs: Any,
) -> AsyncIterator[tuple[LLMResponseProtocol, LLMStreamingChunkProtocol]]:
    """
    Generic streaming structured output helper.

    Yields tuples of (final_response, chunk) so that callers can still
    stream tokens while eventually receiving a fully parsed Pydantic model.

    Parameters
    ----------
    client : Any
        Client with a streaming ``create_chat`` implementation.
    messages : list[dict] or str
        Input messages.
    response_model : type[pydantic.BaseModel]
        Target Pydantic model.
    model : str, optional
        Model identifier.
    **kwargs
        Forwarded to the client's streaming chat method (must include ``stream=True``).

    Yields
    ------
    tuple[LLMResponseProtocol, LLMStreamingChunkProtocol]
        The (eventual) parsed response together with each streaming chunk.
    """
    async for item in client.create_chat(
        messages=messages,
        model=model,
        response_model=response_model,
        stream=True,
        **kwargs,
    ):
        yield item
