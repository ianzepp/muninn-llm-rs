# muninn-llm

LLM-facing syscalls for the Muninn kernel family.

`muninn-llm` provides two `Syscall` implementations for `muninn-kernel`:

- **`LlmSyscall`** — handles `llm:chat` requests with streaming provider deltas (Anthropic and OpenAI)
- **`RoomSyscall`** — manages in-memory rooms with actors, conversation history, and a ReAct tool loop

The crate is intended to be embedded by a Muninn host process that loads config, constructs the syscall handlers, and registers them with `muninn-kernel`.

## Installation

Add to your `Cargo.toml` as a git dependency:

```toml
[dependencies]
llm = { package = "muninn-llm", git = "https://github.com/ianzepp/muninn-llm-rs" }
```

Or with the full package name:

```toml
[dependencies]
muninn-llm = { git = "https://github.com/ianzepp/muninn-llm-rs" }
```

Pin to a specific tag or commit rather than tracking `main`:

```toml
[dependencies]
llm = { package = "muninn-llm", git = "https://github.com/ianzepp/muninn-llm-rs", tag = "v0.1.0" }
```

## Architecture Overview

```
Host Process
  │
  ▼ load_config("llm.toml")
ConfigFile
  │
  ├──▶ LlmSyscall::new(config)  →  register with Kernel as "llm"
  └──▶ RoomSyscall::new(config)  →  register with Kernel as "room"

Kernel routing:
  "llm:chat"     →  LlmSyscall  →  spawns per-request task  →  streams Item frames
  "room:join"    →  RoomSyscall  →  worker task per room
  "room:part"    →  RoomSyscall  →  worker task per room
  "room:history" →  RoomSyscall  →  worker task per room
  "room:list"    →  RoomSyscall  →  direct response
  "room:message" →  RoomSyscall  →  worker task  →  ReAct loop  →  streams Item frames
```

## Configuration

Configuration is loaded from a TOML file with named LLM profiles:

```toml
active = "default"
traits = ["ego/senior", "filter/slack"]

[configs.default]
provider = "anthropic"
model = "claude-sonnet-4-6"
max_tokens = 8096
api_key_env = "ANTHROPIC_API_KEY"
self_prompt = "You are a helpful assistant."

[configs.hand]
provider = "openai"
model = "gpt-4o"
max_tokens = 4096
api_key_env = "OPENAI_API_KEY"
self_prompt = "You are a specialized worker."
```

| Field | Required | Description |
|---|---|---|
| `provider` | yes | `"anthropic"` or `"openai"` |
| `model` | yes | Model identifier (e.g. `"claude-sonnet-4-6"`, `"gpt-4o"`) |
| `max_tokens` | yes | Maximum response tokens |
| `api_key_env` | no | Environment variable holding the API key (default: `ANTHROPIC_API_KEY`) |
| `api_key_value` | no | Literal API key (local overrides only, never commit) |
| `self_prompt` | yes | Persistent identity text for the system prompt |
| `openai_base_url` | no | Custom base URL for OpenAI-compatible APIs |

## LlmSyscall

Handles `llm:chat` — streaming chat completion with history, context, and optional tools.

### Request Frame

```rust
Frame::request("llm:chat") with data:
{
    "config": "default",           // profile name from config file
    "history": [...],              // room conversation history
    "context": [...],              // in-progress tool loop turns (optional)
    "memory": "...",               // long-term memory text (optional)
    "tools": [...]                 // tool definitions (optional)
}
```

### Response Stream

Each request produces a sequence of `Item` frames followed by a terminal `Done` or `Error`:

```
Item { type: "text_delta",     text: "..." }
Item { type: "thinking_delta", thinking: "..." }
Item { type: "tool_use_delta", id, name, input }
Item { type: "done",           stop_reason, model, input_tokens, output_tokens }
Done
```

### Usage

```rust
use std::sync::Arc;
use muninn_kernel::Kernel;
use muninn_llm::config::load_config;
use muninn_llm::LlmSyscall;

let config = load_config(Path::new("llm.toml"))?;
let llm = LlmSyscall::new(config)?;

let mut kernel = Kernel::new();
kernel.register_call(Arc::new(llm));
```

## RoomSyscall

Manages in-memory rooms with actors, conversation history, and a ReAct tool loop.

### Verbs

| Verb | Description | Creates room? |
|---|---|---|
| `room:join` | Add an actor to a room | yes |
| `room:part` | Remove an actor (deletes room when empty) | no |
| `room:history` | Stream conversation history entries | no |
| `room:list` | Stream active room names | n/a |
| `room:message` | User message triggers ReAct loop for all actors | yes |

### room:join

```rust
Frame::request("room:join") with data:
{
    "room": "my-room",
    "actor_name": "alice",
    "config": "default"          // references a named LLM profile
}
```

### room:message

```rust
Frame::request("room:message") with data:
{
    "room": "my-room",
    "from": "user-42",
    "content": "Summarize this document",
    "tools": [...],              // tool definitions (optional)
    "tool_prefixes": ["vfs"]     // allowed tool call prefixes (optional)
}
```

Response stream:

```
Item { type: "text_delta", actor: "alice", room: "my-room", text: "..." }
Item { type: "reply",      actor: "alice", room: "my-room", content: "full reply" }
Done
```

### room:history

```rust
Frame::request("room:history") with data:
{
    "room": "my-room",
    "limit": 50                  // optional, most recent N entries
}
```

Each history entry is streamed as an `Item`:

```
Item { id, ts, from, content, kind }
```

### Usage

```rust
use std::sync::Arc;
use muninn_kernel::Kernel;
use muninn_llm::config::load_config;
use muninn_llm::{LlmSyscall, RoomSyscall};

let config = load_config(Path::new("llm.toml"))?;
let llm = LlmSyscall::new(config.clone())?;
let room = RoomSyscall::new(config);

let mut kernel = Kernel::new();
kernel.register_call(Arc::new(llm));
kernel.register_call(Arc::new(room));
let _handle = kernel.start();
```

## ReAct Tool Loop

When `room:message` is processed, each actor in the room runs a ReAct (Reasoning and Acting) loop:

1. Call `llm:chat` with conversation history, accumulated context, and tool definitions
2. Collect streaming deltas, forward `text_delta` items upstream for low-latency display
3. Check the LLM's `stop_reason`:
   - `end_turn` / `max_tokens` — emit the final reply, done
   - `tool_use` — dispatch each tool call, append results to context, loop back to step 1
4. After 20 rounds, return an error

### Tool Call Security

- Tool calls are checked against the caller-supplied `tool_prefixes` allowlist
- `room:*` write calls are blocked to prevent deadlocks
- `room:history` and `room:list` are allowed (read-only, no deadlock risk)
- `room:delegate` is intercepted and runs a nested actor loop inline

### Side-Channel Events

During the ReAct loop, the room emits fire-and-forget events via the kernel:

| Event | Description |
|---|---|
| `door:thought` | Actor's thinking/reasoning block |
| `door:tool` | Tool call with syscall name and arguments |
| `door:chat` | Actor's final text reply |

## System Prompt Assembly

The system prompt is assembled from 10 semantic slots, ordered from most stable to most behavioral:

| Slot | Name | Source |
|---|---|---|
| 0 | Identity | `self_prompt` from config profile |
| 1 | Commandments | Built-in safety constraints |
| 2 | Context | Role-specific text derived from config name |
| 3 | Tools | Available tool names and usage instructions |
| 4 | Description | Room description (caller-provided) |
| 5 | Notes | Room notes (caller-provided) |
| 7 | Environment | OS info and room name |
| 8 | Memory | Long-term memory (host-provided) |
| 9 | Tone | Composable personality traits |

The prompt is cacheable — it changes only when context fields change, not on every request.

### Built-in Traits

Traits are composable tone modifiers specified in the config file:

| Trait | Effect |
|---|---|
| `ego/intern` | Humble, asks clarifying questions |
| `ego/senior` | Confident and direct |
| `ego/10x` | Extremely confident, volunteers strong opinions |
| `filter/hr` | Professional and workplace-appropriate |
| `filter/slack` | Casual, friendly team chat tone |
| `filter/anon` | Blunt and direct, no professional filter |
| `fever/mild` | Slightly more energetic |
| `fever/hot` | Very energetic and fast-paced |
| `collab/normie` | Collaborative, asks for consensus |
| `collab/feral` | Autonomous, asks only when truly blocked |

## Error Types

```rust
// LlmError — LLM subsystem errors with error codes and retryability
LlmError::ConfigLoad(msg)           // E_CONFIG_LOAD
LlmError::ConfigParse(msg)          // E_CONFIG_PARSE
LlmError::MissingApiKey { var }     // E_MISSING_API_KEY
LlmError::UnknownConfig { name }    // E_UNKNOWN_CONFIG
LlmError::ApiRequest(msg)           // E_API_REQUEST     (retryable)
LlmError::ApiResponse { status }    // E_API_RESPONSE    (retryable for 429, 5xx)
LlmError::ApiParse(msg)             // E_API_PARSE
LlmError::StreamDecode(msg)         // E_STREAM_DECODE

// RoomError — room state errors
RoomError::RoomNotFound { room }        // E_ROOM_NOT_FOUND
RoomError::RoomBusy { room }            // E_ROOM_BUSY
RoomError::ActorNotFound { room, name } // E_ACTOR_NOT_FOUND
RoomError::ActorAlreadyJoined { .. }    // E_ACTOR_ALREADY_JOINED
```

Both error types implement `muninn_kernel::ErrorCode` for automatic frame serialization.

## Module Reference

| Module | Purpose |
|---|---|
| `config` | `ConfigFile`, `LlmProfile`, `load_config()`, `resolve_api_key()` |
| `error` | `LlmError`, `RoomError` with `ErrorCode` implementations |
| `types` | `ContentBlock`, `ContentDelta`, `Message`, `ChatRequest`, `ChatResponse`, `Tool` |
| `prompt_bundle` | `Bundle`, `Slot`, `PromptContext`, `build_system_prompt()` |
| `llm` | `LlmSyscall` — `llm:chat` handler |
| `room` | `RoomSyscall` — `room:*` handlers and ReAct tool loop |
| `room::state` | `Room`, `Actor`, `HistoryEntry`, `HistoryKind`, `ToolOutcomeRecord` |

## Providers

| Provider | API | Streaming | Notes |
|---|---|---|---|
| Anthropic | Messages API | SSE | Native tool use, thinking blocks |
| OpenAI | Chat Completions | NDJSON | System message injection, tool use |

Custom OpenAI-compatible endpoints are supported via `openai_base_url` in the config.

## Status

The API is early-stage. Pin to a tag or revision rather than tracking a moving branch.
