# muninn-llm-rs — Implementation Plan

## What This Crate Is

`muninn-llm` is a Muninn family crate that implements the **LLM tool/context/ReAct/room
loop** as a pair of `Syscall` handlers registered with `muninn-kernel`. It speaks the
muninn frame protocol natively. Host binaries register `LlmSyscall` and `RoomSyscall`
with the kernel builder; everything else flows through frames.

Reference implementations: `prior/kernel/src/llm/` and `prior/kernel/src/room/`.

---

## Decisions

1. **Single crate** — `llm:` and `room:` prefixes live together. Room depends on llm;
   splitting them into two crates would create a one-way dependency between tiny packages.

2. **Memory slot exists, not populated** — `PromptBundle` has a slot 8 (`Memory`) and
   the `PromptContext` struct accepts an optional memory string, but `muninn-llm` does not
   make any outbound `ems:` calls to fill it. The host may pre-populate memories before
   calling `room:message`, or leave the slot empty.

3. **Tool allowlist is caller-supplied** — `prior/` hardcodes `["vfs:", "ems:", "exec:"]`.
   Here, the allowed tool prefixes are passed as part of the `room:join` / `room:message`
   request data so each client project configures what its room may call.

4. **Streaming** — Both `llm:chat` and `room:message` emit incremental `Item` frames as
   content arrives from the provider, rather than buffering the full response. Text delta
   blocks are emitted as `Item` frames; a final `Done` frame closes the stream.

---

## Syscall Surface

### `llm:` prefix — `LlmSyscall`

| Verb     | Request data                              | Response frames                              |
|----------|-------------------------------------------|----------------------------------------------|
| `chat`   | `ChatRequest` (config, history, context, tools) | `Item` per content delta, then `Done`   |
| `models` | `{ config: String }`                      | `Item` per available model name, then `Done` |

`llm:chat` is the raw provider call. It does not run a tool loop. Room calls this
repeatedly as part of the ReAct loop.

### `room:` prefix — `RoomSyscall`

| Verb       | Request data                                     | Response frames                             |
|------------|--------------------------------------------------|---------------------------------------------|
| `message`  | `{ room, from, content, tool_prefixes? }`        | `Item` per actor reply delta, then `Done`   |
| `join`     | `{ room, actor: { name, config } }`              | `Done`                                      |
| `part`     | `{ room, actor: name }`                          | `Done`                                      |
| `history`  | `{ room, limit? }`                               | `Item` per `HistoryEntry`, then `Done`      |
| `list`     | `{}`                                             | `Item` per room name, then `Done`           |
| `delegate` | `{ role, prompt, tool_prefixes? }`               | `Item` per delegate reply delta, then `Done`|

`room:message` is the full ReAct entry point. `room:delegate` is intercepted inside the
tool loop to run a nested actor inline, avoiding deadlock through the room worker mailbox.

---

## Streaming Design

`prior/` defers streaming (single item + done). This crate streams from the start.

### Provider streaming

Both `AnthropicClient` and `OpenAiClient` consume the provider's SSE/NDJSON stream via
`reqwest` with `.bytes_stream()`. Each delta event is decoded and emitted as a
`ContentDelta` to the caller.

```
Provider SSE stream
  └─ per delta event:
       AnthropicClient / OpenAiClient → ContentDelta
         ContentDelta::TextDelta(String)
         ContentDelta::ThinkingDelta(String)
         ContentDelta::ToolUseDelta { id, name, input_json_fragment }
         ContentDelta::Done { stop_reason, model, input_tokens, output_tokens }
```

### Frame streaming from llm:chat

`LlmSyscall::handle_chat` translates provider deltas to frames:

```
ContentDelta::TextDelta(s)         → Item { type: "text_delta", text: s }
ContentDelta::ThinkingDelta(s)     → Item { type: "thinking_delta", thinking: s }
ContentDelta::ToolUseDelta { .. }  → Item { type: "tool_use_delta", id, name, input }
ContentDelta::Done { .. }          → Item { type: "done", stop_reason, model, usage }
                                   → Done (terminal)
```

The room loop collects `llm:chat` items from the stream and reconstructs full
`ContentBlock` values (assembling tool_use input JSON from fragments) before running
tool dispatch. This keeps the frame protocol clean while enabling streaming text output
to downstream clients.

### Frame streaming from room:message

`RoomSyscall` streams actor reply text through to the caller as it arrives:

```
room:message request
  └─ for each actor in parallel:
       run_actor_loop (streaming)
         └─ each TextDelta from llm:chat
              → Item { type: "text_delta", actor, room, text }
         └─ each ToolUse block (reconstructed)
              → Item { type: "tool_call", actor, room, syscall, data }
         └─ final text assembled
              → Item { type: "reply", actor, room, content }
  └─ Done
```

Broadcast frames (`door:thought`, `door:tool`, `door:chat`) are still emitted
fire-and-forget via `caller.spam()` for connected gateway subscribers.

---

## File Layout

```
muninn-llm-rs/
  Cargo.toml
  PLAN.md
  src/
    lib.rs              — pub re-exports: LlmSyscall, RoomSyscall, types
    error.rs            — LlmError (thiserror enum, ErrorCode impl)
    types.rs            — ContentBlock, ContentDelta, Content, Tool, Message,
                          ChatRequest, ChatResponse, from_data, to_data
    config.rs           — ConfigFile, LlmProfile, load_runtime_config,
                          load_config(path), resolve_api_key
    prompt_bundle.rs    — PromptContext, build_system_prompt (10-slot assembly)
    client_anthropic.rs — AnthropicClient: SSE stream → ContentDelta
    client_openai.rs    — OpenAiClient: NDJSON stream → ContentDelta
    llm/
      mod.rs            — LlmSyscall (Syscall impl, prefix = "llm"),
                          run loop, dispatch by verb
      chat.rs           — handle_chat: stream deltas → Item frames,
                          history_to_messages, chrono gap markers
    room/
      mod.rs            — RoomSyscall (Syscall impl, prefix = "room"),
                          in-memory room map, run loop, dispatch by verb
      state.rs          — Room, Actor, HistoryEntry, HistoryKind
      message.rs        — run_actor_loop (streaming), dispatch_tools,
                          dispatch_one_tool, dispatch_delegate (Box::pin),
                          reconstruct_content_blocks
      handlers.rs       — handle_join, handle_part, handle_history, handle_list
```

---

## Core Types

### `error.rs`

```rust
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    ConfigLoad(String),
    ConfigParse(String),
    MissingApiKey { var: String },
    UnknownConfig { name: String },
    ApiRequest(String),
    ApiResponse { status: u16, body: String },
    ApiParse(String),
    StreamDecode(String),
    Deserialize(String),
    Serialize(String),
    InternalCall(String),
    HttpClientBuild(String),
    PipeSend(String),
}
// impl ErrorCode for LlmError
```

### `types.rs`

```rust
// Content blocks (provider-neutral)
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: String, is_error: Option<bool> },
    Thinking { thinking: String },
    Unknown,
}

// Streaming delta from provider
pub enum ContentDelta {
    TextDelta(String),
    ThinkingDelta(String),
    ToolUseDelta { id: String, name: String, input_fragment: String },
    Done { stop_reason: String, model: String, input_tokens: u64, output_tokens: u64 },
}

pub enum Content { Text(String), Blocks(Vec<ContentBlock>) }

pub struct Tool { pub name: String, pub description: String, pub input_schema: Value }
pub struct Message { pub role: String, pub content: Content }

pub struct ChatRequest {
    pub config: String,
    pub history: Vec<HistoryEntry>,       // from room::state
    pub context: Vec<Message>,            // in-progress tool loop turns
    pub tools: Option<Vec<Tool>>,
}

pub struct ChatResponse {
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}
```

### `config.rs`

```rust
pub struct LlmProfile {
    pub provider: String,          // "anthropic" | "openai"
    pub model: String,
    pub max_tokens: u32,
    pub api_key_env: Option<String>,
    pub api_key: Option<String>,   // literal (not recommended)
    pub openai_base_url: Option<String>,
    pub self_prompt: String,
}

pub struct ConfigFile {
    pub active: String,
    pub configs: HashMap<String, LlmProfile>,
    pub trait_names: Vec<String>,
}
```

TOML format:
```toml
active = "default"
traits = ["concise", "direct"]

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

### `prompt_bundle.rs`

10-slot assembly. Slots are filled in order; empty slots are skipped with no separator.

```rust
pub struct PromptContext<'a> {
    pub config: &'a str,          // profile name (used for slot 0 identity)
    pub self_prompt: &'a str,     // slot 0: identity
    pub tools: Option<&'a [Tool]>,// slot 3: tools primary reference
    pub room: &'a str,            // slot 7: environment
    pub description: &'a str,     // slot 4: room description
    pub notes: &'a str,           // slot 5: room notes
    pub memory: &'a str,          // slot 8: left empty by this crate; caller may pre-fill
    pub traits: &'a [String],     // slot 9: tone/personality
}

pub fn build_system_prompt(ctx: &PromptContext<'_>) -> String
```

Slots:
- 0 Identity — `self_prompt` value
- 1 Commandments — static safety text (hardcoded in crate, not user-configurable)
- 2 Context — role-specific (derived from `config` name)
- 3 ToolsPrimary — rendered tool list if tools provided, else empty
- 4 Description — `description` (room description.md content or empty)
- 5 Notes — `notes` (room notes.md content or empty)
- 6 *(reserved)*
- 7 Environment — current room name, UTC timestamp
- 8 Memory — `memory` string (empty unless host pre-fills)
- 9 Tone — comma-joined `traits` list

### `room/state.rs`

```rust
pub struct Actor {
    pub name: String,
    pub config: String,  // references a ConfigFile profile
}

pub enum HistoryKind { User, Assistant, System }

pub struct HistoryEntry {
    pub id: u64,
    pub ts: i64,          // unix seconds
    pub from: String,
    pub content: String,
    pub kind: HistoryKind,
}

pub struct Room {
    pub actors: Vec<Actor>,
    pub history: Vec<HistoryEntry>,
    pub next_id: u64,
}

impl Room {
    pub fn add_message(&mut self, from: &str, content: &str, kind: HistoryKind)
    pub fn trim_history(&mut self, max: usize)   // keep last N entries
}
```

---

## ReAct Loop (streaming)

`room/message.rs` — `run_actor_loop`:

```
run_actor_loop(caller, config, history, tools, allowed_prefixes, room, actor_name, actors)
  └─ context: Vec<Message> = []   // accumulates tool turns across rounds
  └─ for round in 0..MAX_TOOL_ROUNDS:
       1. build_chat_data(config, history, context, tools)
       2. Frame::request("llm:chat", chat_data).with_room(room)
       3. caller.call(chat_frame) → stream
       4. collect stream, reconstruct ContentBlocks from deltas:
            - stream Item { type: "text_delta" }     → buffer text
            - stream Item { type: "thinking_delta" } → buffer thinking
            - stream Item { type: "tool_use_delta" } → buffer tool_use
            - stream Item { type: "done" }           → extract stop_reason
          *While buffering, forward text_delta/thinking_delta upstream as Item frames*
       5. check stop_reason:
            "end_turn" | "max_tokens":
              → emit Item { type: "reply", actor, content: buffered_text }
              → return Ok(buffered_text)
            "tool_use":
              → append assistant turn (with tool_use blocks) to context
              → dispatch_tools(caller, tool_blocks, actors, allowed_prefixes, room)
              → append user turn (tool_result blocks) to context
              → continue
            other → Err("unknown stop_reason")
  └─ Err("exceeded MAX_TOOL_ROUNDS")
```

`MAX_TOOL_ROUNDS = 20` (same as `prior/`).

### Tool dispatch

`dispatch_one_tool`:

1. Extract `syscall` from `input.syscall`.
2. Intercept `room:delegate` → `dispatch_delegate` (inline nested actor loop).
3. Allow `room:history` and `room:list` (read-only, no deadlock risk).
4. Check `syscall` prefix is in `allowed_prefixes` (caller-supplied list). Block others.
5. Build `Frame::request(syscall, data_from_input)`.
6. `caller.call(req)` → collect → format as JSON-lines string → `ToolResult`.

### Delegation

`dispatch_delegate`:

1. Extract `role` and `prompt` from tool input.
2. Find actor in room whose `config == role`.
3. Build single-entry history `[HistoryEntry { kind: User, content: prompt }]`.
4. `Box::pin(run_actor_loop(...))` — breaks async recursion cycle.
5. Return delegate's text reply as `ToolResult`.

---

## Provider Clients

### Shared trait (internal, not public)

```rust
// internal to this crate — not pub, two impls only
trait ProviderClient: Send + Sync {
    async fn stream_chat(
        &self,
        model: &str,
        max_tokens: u32,
        system: &str,
        messages: &[Message],
        tools: Option<&[Tool]>,
    ) -> Result<BoxStream<'static, Result<ContentDelta, LlmError>>, LlmError>;
}
```

Using `futures_util::stream::BoxStream` for the return type.

### `AnthropicClient`

- Endpoint: `https://api.anthropic.com/v1/messages`
- Headers: `x-api-key`, `anthropic-version`, `anthropic-beta: interleaved-thinking-2025-05-14`
- Request: `{ model, max_tokens, system, messages, tools?, stream: true }`
- SSE events decoded:
  - `content_block_start` with `type: "text"` / `type: "tool_use"` / `type: "thinking"`
  - `content_block_delta` with `delta.type: "text_delta"` / `"thinking_delta"` / `"input_json_delta"`
  - `message_delta` with `stop_reason`, usage
  - `message_stop` → emit `ContentDelta::Done`

### `OpenAiClient`

- Endpoint: configurable base URL (defaults to `https://api.openai.com`)
- Path: `/v1/chat/completions`
- Headers: `Authorization: Bearer <key>`
- Request: `{ model, max_tokens, messages (with system as first), tools?, stream: true }`
- NDJSON `data:` lines decoded:
  - `choices[0].delta.content` → `ContentDelta::TextDelta`
  - `choices[0].delta.tool_calls[].function.arguments` → `ContentDelta::ToolUseDelta`
  - `choices[0].finish_reason` → `ContentDelta::Done`

Both clients map their wire formats into the same `ContentDelta` enum so `chat.rs` is
provider-agnostic.

---

## LlmSyscall — run loop

`llm/mod.rs`:

```
LlmSyscall::new(pipe: PipeEnd, config: ConfigFile)
  └─ builds HashMap<config_name, Box<dyn ProviderClient>>

LlmSyscall::run(mut self):
  └─ inflight: HashMap<Uuid, JoinHandle<()>>
  └─ while let Some(frame) = pipe.recv():
       Status::Cancel → abort inflight task by parent_id
       syscall verb "chat"   → spawn handle_chat task
       syscall verb "models" → spawn handle_models task
       unknown               → send Error frame
```

Each request spawns its own task so concurrent rooms (multiple actors, multiple users)
make overlapping API calls without head-of-line blocking.

---

## RoomSyscall — run loop

`room/mod.rs`:

```
RoomSyscall::new(pipe: PipeEnd, config: ConfigFile)
  └─ rooms: HashMap<String, Room>  (in-memory only)

RoomSyscall::run(mut self):
  └─ while let Some(frame) = pipe.recv():
       "message"  → run_actor_loop per actor (tokio::spawn, fan-in results)
       "join"     → add actor to room (create room if absent)
       "part"     → remove actor from room (delete room if empty)
       "history"  → stream HistoryEntry items
       "list"     → stream room names
       "delegate" → (only reached from within tool loop via direct call, not frame)
```

Room state lives in the `RoomSyscall` task — no shared state, no mutex. All mutation
goes through the single-task message loop. Concurrent `room:message` requests for the
*same* room are queued naturally by the mpsc channel.

---

## Integration Example

```rust
use muninn_kernel::{Kernel, PipeEnd};
use muninn_llm::{LlmSyscall, RoomSyscall, ConfigFile};

let config = muninn_llm::config::load_runtime_config();

let (llm_pipe, llm_end)   = PipeEnd::pair();
let (room_pipe, room_end) = PipeEnd::pair();

let llm_syscall  = LlmSyscall::new(llm_end, config.clone())?;
let room_syscall = RoomSyscall::new(room_end, config)?;

let kernel = Kernel::builder()
    .register_call(llm_syscall)
    .register_call(room_syscall)
    .build();

kernel.start();

// Join a room, then send a message
let sender = kernel.sender();
sender.send(Frame::request("room:join", data!{
    "room"   => "my-room",
    "actor"  => { "name": "assistant", "config": "default" }
})).await?;

let stream = kernel.call(Frame::request("room:message", data!{
    "room"            => "my-room",
    "from"            => "user",
    "content"         => "Hello!",
    "tool_prefixes"   => ["vfs:", "ems:"]
})).await?;

while let Some(frame) = stream.next().await {
    // Item { type: "text_delta", actor: "assistant", text: "..." }
    // Item { type: "reply", actor: "assistant", content: "..." }
    // Done
}
```

---

## Implementation Phases

### Phase 1 — Skeleton

- [ ] `src/lib.rs` — pub re-exports
- [ ] `src/error.rs` — `LlmError`, `ErrorCode` impl
- [ ] `src/types.rs` — `ContentBlock`, `ContentDelta`, `Content`, `Tool`, `Message`,
       `ChatRequest`, `ChatResponse`, `from_data`, `to_data`
- [ ] `src/config.rs` — `ConfigFile`, `LlmProfile`, loaders, `resolve_api_key`
- [ ] `src/room/state.rs` — `Room`, `Actor`, `HistoryEntry`, `HistoryKind`

### Phase 2 — Prompt bundle

- [ ] `src/prompt_bundle.rs` — `PromptContext`, `build_system_prompt`, 10-slot assembly
- [ ] Unit tests for slot ordering and empty-slot elision

### Phase 3 — Provider clients (streaming)

- [ ] `src/client_anthropic.rs` — SSE stream → `ContentDelta`
- [ ] `src/client_openai.rs` — NDJSON stream → `ContentDelta`
- [ ] Integration tests against recorded fixtures (no live network in CI)

### Phase 4 — LLM syscall

- [ ] `src/llm/chat.rs` — `handle_chat`: delta stream → Item frames,
       `history_to_messages`, chrono gap markers
- [ ] `src/llm/mod.rs` — `LlmSyscall`, run loop, task fan-out, cancellation
- [ ] Tests: mock provider, verify frame sequence

### Phase 5 — Room syscall

- [ ] `src/room/handlers.rs` — `handle_join`, `handle_part`, `handle_history`,
       `handle_list`
- [ ] `src/room/message.rs` — `run_actor_loop` (streaming), `dispatch_tools`,
       `dispatch_one_tool`, `dispatch_delegate`, `reconstruct_content_blocks`
- [ ] `src/room/mod.rs` — `RoomSyscall`, run loop
- [ ] Tests: full ReAct loop with mock `llm:chat` responder

### Phase 6 — Integration

- [ ] End-to-end test: kernel + LlmSyscall + RoomSyscall + mock provider
- [ ] Verify streaming frame sequence for text, tool_use, delegation
- [ ] Verify `door:thought`, `door:tool`, `door:chat` broadcast emission

### Phase 7 — Hygiene

- [ ] `cargo clippy -- -D warnings`
- [ ] `cargo test`
- [ ] No `unwrap()`, `expect()`, `panic!()` in non-test code
- [ ] No `#[allow(dead_code)]`
- [ ] File sizes ≤ 400 lines, function sizes ≤ 60 lines (per `prior/` standards)

---

## Standards (from prior/CLAUDE.md, adopted here)

- No panics in non-test code. Use `?`, `let … else`, never `unwrap()`.
- No globals. All dependencies passed explicitly.
- Guard clauses over nesting. Happy path at left margin.
- Channels over mutexes for cross-task state.
- `thiserror` for all error types. `anyhow` only in `main()`.
- Import groups: std → external → internal, blank line between.
- Owned types across async boundaries (`String` over `&str`).
- Flat modules until a module genuinely needs submodules.
- Concrete types over trait objects; `dyn` only at true polymorphism boundaries.
- No silent loss: no `let _ =` on meaningful errors, no `.ok()` hiding failures.
