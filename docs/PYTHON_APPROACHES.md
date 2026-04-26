# Python Approaches in the ai_api Project

This document explains the key design patterns and language features used throughout the `ai_api` project. It is written for a reader with intermediate Python knowledge who wants to understand *why* certain choices were made and how they help us build a clean, extensible library for working with multiple LLM providers (Ollama, xAI, and future additions).

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Python Approaches in the ai_api Project](#python-approaches-in-the-ai_api-project)
  - [Design Philosophy](#design-philosophy)
    - [Provider-Specific Data Objects](#provider-specific-data-objects)
    - [Generic Protocols for Common Interfaces](#generic-protocols-for-common-interfaces)
    - [The Inherent Tension in Protocols](#the-inherent-tension-in-protocols)
  - [Summary of Approaches Used](#summary-of-approaches-used)
    - [Background](#background)
    - [How They Help This Project](#how-they-help-this-project)
  - [Decorators in Detail](#decorators-in-detail)
    - [Popular Decorators](#popular-decorators)
    - [The Decorators Used](#the-decorators-used)
      - [`@dataclass(frozen=True)`](#dataclassfrozentrue)
      - [`@property`](#property)
      - [`@classmethod`](#classmethod)
      - [`@runtime_checkable`](#runtime_checkable)
  - [Protocols vs Abstract Base Classes (ABC)](#protocols-vs-abstract-base-classes-abc)
    - [Protocols (Structural Subtyping)](#protocols-structural-subtyping)
    - [Abstract Base Classes (ABC)](#abstract-base-classes-abc)
    - [Comparison](#comparison)
  - [Pydantic for Structuring Data](#pydantic-for-structuring-data)
    - [What Problem Does Pydantic Solve?](#what-problem-does-pydantic-solve)
    - [Alternative Approaches](#alternative-approaches)
    - [Comparison with `db_responses_schema.py`](#comparison-with-db_responses_schemapy)
  - [Advanced Typing & Data Modelling Patterns](#advanced-typing--data-modelling-patterns)
    - [Mixing `dataclasses.field` and `pydantic.Field` (Often in the Same File)](#mixing-dataclassesfield-and-pydanticfield-often-in-the-same-file)
    - [Using `Self` from the `typing` Library](#using-self-from-the-typing-library)
    - [String Forward References (`"OllamaResponse | str"`)](#string-forward-references-ollamaresponse--str)
  - [Additional Notes](#additional-notes)

<!-- markdown-toc end -->

---

## Design Philosophy

The core challenge in this project is supporting multiple LLM providers while keeping the rest of the system (persistence, logging, client factories, etc.) as generic as possible.

### Provider-Specific Data Objects

Each provider has its own way of representing requests and responses:

- **Ollama** uses its native HTTP API with fields like `keep_alive`, `done_reason`, `total_duration`, and a specific message format.
- **xAI** uses the official SDK with different structures for multimodal input, tool calls, and usage statistics.

Rather than forcing everything into one rigid shape (which would be painful), the allows **provider-specific data classes** (`OllamaRequest`, `xAIRequest`, `OllamaResponse`, `xAIResponse`, etc.). These live in `data_structures/ollama_objects.py` and `data_structures/xai_objects.py`.

### Generic Protocols for Common Interfaces

These provider-specific objects are then brought together through **generic protocols** defined in `data_structures/base_objects.py`:

- `LLMRequestProtocol`
- `LLMResponseProtocol`
- `LLMStreamingChunkProtocol`

Each protocol requires only three methods: `meta()`, `payload()`, and `endpoint()`. Because these methods return plain dictionaries (or `LLMEndpoint`), the persistence layer, logging, and client factory can work with *any* provider without knowing its internal details.

This is the heart of the design: **specialise where the providers differ, standardise where the rest of the system needs consistency**.

### The Inherent Tension in Protocols

There is a natural tension in this approach:

- **Sometimes we need strong uniformity.**  
  The persistence system must be able to call `persist_request()` and `persist_response()` on objects from *any* provider and always get the same shape of data. This is why `LLMRequestProtocol` and `LLMResponseProtocol` are strict — every implementation *must* provide `meta()`, `payload()`, and `endpoint()`.

- **Sometimes we need flexibility.**  
  The `create_chat()` method on client classes is deliberately *not* uniform. A turn-based chat, a streaming chat, and a batch chat have fundamentally different signatures and return types. Forcing them to look identical would make the code more complex and less natural to use.

The `create_chat()` method is therefore defined very loosely (using `*args, **kwargs` and returning `Any`). This is the pragmatic choice for the user-facing API. The strict guarantees are applied *downstream* on the objects that actually get persisted.

This tension — strict protocols for data, flexible methods for control flow — is a deliberate and recurring theme in the project.

---

## Summary of Approaches Used

### Background

Modern Python has evolved a rich set of tools for writing clean, maintainable, and type-safe code without the verbosity of older object-oriented patterns. Three concepts are particularly important in this project:

- **Decorators** (since Python 2.4, greatly expanded in 3.x)
- **Abstract Base Classes (ABC)** (introduced in Python 2.6, `abc` module)
- **Protocols** (structural subtyping, formalised in Python 3.8 via PEP 544)

These features allow us to write code that is both flexible and self-documenting.

### How They Help This Project

| Approach    | Primary Use in ai_api                                                                       | Benefit                                                                                     |
|-------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Decorators  | `@dataclass(frozen=True)`, `@property`, `@classmethod`, `@runtime_checkable`                | Clean immutable data, computed properties, class-level behaviour, runtime protocol checking |
| Protocols   | `LLMRequestProtocol`, `LLMResponseProtocol`, `LLMProviderAdapter`                           | Duck typing with static type checking — no inheritance required                             |
| ABC         | (Used lightly) `LLMProviderAdapter` was originally an ABC before Protocols were implemented | Nominal inheritance when we want explicit "is-a" relationships                              |
| Pydantic    | `BaseModel` inheritance in request/response objects                                         | Runtime validation + excellent editor support                                               |
| Dataclasses | `OllamaRequest`, `xAIRequest`, `LLMEndpoint`                                                | Simple, immutable data containers with generated `__init__`, `__repr__`, etc.               |
|             |                                                                                             |                                                                                             |

---

## Decorators in Detail

Decorators are functions that take another function (or class) and return a modified version. They are applied with the `@` syntax and are executed at definition time.

### Popular Decorators

- `@staticmethod` / `@classmethod`
- `@property` / `@setter`
- `@functools.lru_cache`, `@functools.wraps`
- `@contextlib.contextmanager`
- `@dataclass` (from `dataclasses` module)

### The Decorators Used

#### `@dataclass(frozen=True)`

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class LLMEndpoint:
    provider: str
    model: str
    extra: Any = field(default_factory=dict)
```

**What it does**: Automatically generates `__init__`, `__repr__`, `__eq__`, and other dunder methods.

**`frozen=True`** makes the instance immutable after creation. This is excellent for data that should never change (requests, responses, endpoints). It prevents accidental mutation bugs and makes objects safe to use as dictionary keys or in sets.

#### `@property`

```python
class xAIResponse:
    @property
    def text(self) -> str:
        if self.choices:
            return self.choices[0].get("message", {}).get("content", "")
        return ""
```

**What it does**: Allows a method to be accessed like an attribute. The method is called every time the attribute is accessed.

The project uses it for derived values (`text`, `tool_calls`) that are computed from the raw response data. It keeps the public interface clean while hiding implementation details.

#### `@classmethod`

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "LLMEndpoint":
    return cls(...)
```

**What it does**: The method receives the *class* as the first argument (`cls`) instead of an instance. It is commonly used for alternative constructors.

In our code it provides a clean way to build objects from dictionaries (useful when loading from JSON or Postgres).

#### `@runtime_checkable`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMRequestProtocol(Protocol):
    def meta(self) -> dict[str, Any]: ...
```

**What it does**: Marks a `Protocol` so that `isinstance(obj, MyProtocol)` works at runtime (normally protocols are only for static type checking).

The project uses it on all our protocols so that the persistence layer can perform lightweight runtime checks if needed, while still getting full static type safety.

---

## Protocols vs Abstract Base Classes (ABC)

### Protocols (Structural Subtyping)

A `Protocol` (PEP 544) defines an interface by *structure* rather than by inheritance. If a class has the required methods/attributes with compatible signatures, it is considered to implement the protocol — even if it never inherited from it.

**Example**:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMRequestProtocol(Protocol):
    def meta(self) -> dict[str, Any]: ...
    def payload(self) -> dict[str, Any]: ...
    def endpoint(self) -> LLMEndpoint: ...
```

Any class that defines those three methods is automatically a `LLMRequestProtocol`. This is called **structural subtyping** (or duck typing with static checking).

**Advantages**:
- No forced inheritance hierarchy
- Extremely flexible for third-party or legacy code
- Perfect when you want "if it walks like a duck..."

### Abstract Base Classes (ABC)

An ABC defines an interface through **nominal subtyping** (inheritance). A class must explicitly inherit from the ABC and implement the abstract methods.

```python
from abc import ABC, abstractmethod

class LLMProviderAdapter(ABC):
    @abstractmethod
    async def create_chat(self, *args, **kwargs): ...
```

**Advantages**:
- Clear "is-a" relationship
- Can provide shared implementation in the base class
- Better when you want to enforce a strict inheritance contract

### Comparison

| Aspect               | Protocol                             | ABC                                                                  |
|----------------------|--------------------------------------|----------------------------------------------------------------------|
| Typing style         | Structural (duck typing)             | Nominal (inheritance)                                                |
| Must inherit?        | No                                   | Yes                                                                  |
| Runtime `isinstance` | Only with `@runtime_checkable`       | Always works                                                         |
| Best for             | Flexible interfaces, data contracts  | Frameworks, plugin systems, "is-a"                                   |
| Our usage            | `LLM*Protocol`, `LLMProviderAdapter` | Originally used for `LLMProviderAdapter` (later changed to Protocol) |

In this project **Protocols** are preferred for most interfaces because they give us maximum flexibility while still providing excellent static type checking. ABC is only needed when we want explicit inheritance or shared implementation.

---

## Pydantic for Structuring Data

### What Problem Does Pydantic Solve?

Pydantic (v2) provides runtime data validation, parsing, and excellent editor support through `BaseModel`. It solves the classic problem of "I have a dictionary from an API — is it valid and what type is everything?"

Key features used:
- Automatic validation on construction
- Type coercion (e.g. string → int)
- Rich error messages
- `model_dump()` / `model_validate()` for serialisation

### Alternative Approaches

| Approach             | Pros                                      | Cons                               | When we might use it           |
|----------------------|-------------------------------------------|------------------------------------|--------------------------------|
| Plain `dataclass`    | Zero dependencies, fast, immutable option | No runtime validation              | Simple internal data           |
| `TypedDict`          | Lightweight, type-checker friendly        | No runtime validation              | When we only need static types |
| `attrs`              | Similar to dataclasses + validators       | Another dependency                 | Legacy codebases               |
| Manual `if` checks   | Full control                              | Verbose, error-prone               | Almost never                   |
| Pydantic `BaseModel` | Validation + great DX                     | Small performance cost, dependency | Our request/response objects   |

### Comparison with `db_responses_schema.py`

The file `data_structures/db_responses_schema.py` defines the raw SQL table structure for Postgres persistence. It is deliberately *not* a Pydantic model — it is a schema definition used by `asyncpg` to create tables and insert data.

**Key differences**:

- **Pydantic models** (`OllamaRequest`, `xAIResponse`, etc.) live in the application layer. They validate and structure *incoming/outgoing* data.
- **Postgres schema** (`db_responses_schema.py`) defines the *storage* layer. It uses raw column types (`TEXT`, `JSONB`, `UUID`, `TIMESTAMPTZ`) and is concerned with database constraints and indexing, not Python objects.

The project deliberately keep these two concerns separate:
- Pydantic handles the "Python world"
- The schema handles the "database world"

The `PersistenceManager` then translates between them using the protocol methods (`meta()`, `payload()`, `endpoint()`), which return plain dictionaries ready for `json.dumps()` or direct insertion into `JSONB` columns.

This separation gives us clean boundaries and makes it easy to swap storage backends in the future.

---

## Advanced Typing & Data Modelling Patterns

This section documents three important patterns used in `data_structures/ollama_objects.py` and `data_structures/xai_objects.py`.

### Mixing `dataclasses.field` and `pydantic.Field` (Often in the Same File)

The project deliberately uses two different field declaration helpers in the same files.

| Helper              | Library       | Context                                                                                    | Why we use it                                                                  |
|---------------------|---------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `dataclasses.field` | `dataclasses` | Lightweight immutable containers (`xAIInput`, `OllamaInput`, message classes)              | Zero overhead, `default_factory` safety, works with `@dataclass(frozen=True)`  |
| `pydantic.Field`    | `pydantic`    | Full request/response models that inherit from `BaseModel` (`xAIRequest`, `OllamaRequest`) | Validation, JSON schema generation, rich metadata (`description`, constraints) |

**Example from `xai_objects.py`:**

```python
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

@dataclass(frozen=True)
class xAIInput:
    messages: tuple[xAIMessage, ...] = field(default_factory=tuple)   # ← dataclass field

class xAIRequest(BaseModel):
    input: xAIInput | str = Field(..., description="Accepts str or xAIInput")  # ← pydantic Field
```

**Why mix them?**  
Pure data holders stay fast and simple. Domain models get full Pydantic power. Small adapter methods (`from_list()`, `to_sdk_chat_kwargs()`, etc.) handle conversion between the two worlds.

**Files using this pattern:**
- `src/ai_api/data_structures/xai_objects.py`
- `src/ai_api/data_structures/ollama_objects.py`

### Using `Self` from the `typing` Library

`Self` (Python 3.11+) is the modern, clean way to make classmethods return the *exact* subclass type they were called on.

**Current usage (after upgrade):**

```python
from typing import Self

@classmethod
def parse_json(cls, json_data: str | bytes | bytearray) -> Self:
    return cls.model_validate_json(json_data)

@classmethod
def from_ollama_response(cls, response: "OllamaResponse | str") -> Self:
    ...
```

This replaced the older `TypeVar("T", bound=...)` pattern.

**Files currently using `Self`:**
- `src/ai_api/data_structures/ollama_objects.py` — `OllamaJSONResponseSpec`
- `src/ai_api/data_structures/xai_objects.py` — `xAIJSONResponseSpec`

### String Forward References (`"OllamaResponse | str"`)

When a method needs to accept either a full response object **or** raw text,  use a string forward reference to avoid circular import problems at class definition time.

**Example:**

```python
@classmethod
def from_ollama_response(cls, response: "OllamaResponse | str") -> Self:
    if isinstance(response, str):
        json_data = response
    else:
        json_data = response.text
    return cls.parse_json(json_data)
```

**Benefits:**
- Prevents "name not defined" errors
- Gives users a convenient API (pass object or just text)
- Still fully type-safe

**Files using this pattern:**
- `src/ai_api/data_structures/ollama_objects.py`
- `src/ai_api/data_structures/xai_objects.py`

---

## Additional Notes

- This project makes heavy use of `async` / `await` throughout because LLM calls are I/O-bound. The client classes and persistence methods are all asynchronous.
- `LLMEndpoint` is a frozen dataclass that can be converted to/from dictionaries — this makes it easy to store in Postgres while still having nice Python objects in memory.
- The project deliberately avoids deep inheritance hierarchies. Most classes are small and focused. Composition + protocols are preferred over inheritance.

</FILE>
