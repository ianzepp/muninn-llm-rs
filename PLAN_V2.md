# muninn-llm-rs — Plan V2

## Purpose

`muninn-llm` is a Rust library for client projects that already want a
message/frame/event architecture and need an embedded LLM loop on top of it.

It is not just a provider wrapper.
It is the layer that:

- normalizes provider chat/tool streams
- runs the room/tool/delegation loop
- emits request/response frames
- emits side-channel room activity events (`door:*`) for client-project code

The crate should remain frame-native and transport-agnostic.

---

## Source Of Truth

The original `PLAN.md` remains useful as prototype history, but it is no longer
the governing architecture document.

This V2 plan is based on:

- the current `muninn-llm` implementation
- the stronger room/door patterns in `../prior`
- the higher-level client adapter ideas in `../abbot`

Primary architectural direction:

1. Keep the current streaming provider design.
2. Restore `door:*` emissions as a first-class library contract.
3. Move room ownership toward per-room workers rather than one global mutex map.
4. Freeze the public frame contracts before adding more features.

---

## Current Status

Implementation status against this plan:

- Phase 1 is effectively complete: the intended public frame surface is now
  documented here and matches the current crate shape.
- Phase 2 is complete: `door:thought`, `door:tool`, and `door:chat` are back in
  place with tests.
- Phase 3 is not complete, but a temporary safety measure now exists:
  same-room `room:message` overlap is blocked by an in-flight guard so causal
  history corruption cannot occur silently.
- Phase 4 is in progress:
  several high-value regression tests now exist, but full room/message
  pipe-level coverage and cancellation integration coverage are still missing.

The most important remaining architectural step is still the per-room worker
refactor.

---

## Validated Decisions

These decisions have now been tested or stress-reviewed strongly enough to
consider them settled unless new evidence appears:

- `door:*` should remain a core library event contract
- provider streaming must remain incremental end-to-end
- synthetic chrono `"user"` turns were the wrong design for this crate
- same-room serialization is required for correctness, not just cleanliness
- malformed tool payloads must fail explicitly instead of silently degrading

---

## Temporary Compromises

The current implementation contains one deliberate stopgap that should not be
mistaken for the target architecture:

- same-room message overlap is currently prevented by a room-level in-flight
  guard that rejects concurrent `room:message` work for the same room

This is acceptable as a correctness barrier, but it is not the intended final
design.

The target remains:

- per-room worker ownership
- explicit same-room scheduling semantics
- cancellation-aware actor lifecycle handling inside the room worker model

---

## Design Decisions

### 1. `door:*` is core, not optional

`door:thought`, `door:tool`, and `door:chat` are part of the intended library
event surface for client projects.

They are informational broadcasts, not part of the synchronous result stream.

Synchronous callers should still use:

- `room:message` response items for the direct request/response path

Client observers should use:

- `door:thought` for model reasoning/thought activity
- `door:tool` for tool call visibility
- `door:chat` for final actor chat output

This follows the successful `prior` pattern.

### 2. Public frame contracts must be stable

The crate needs explicit, versioned-in-practice request/response shapes.

The current public surface should be:

- `llm:chat`
- `room:join`
- `room:part`
- `room:history`
- `room:list`
- `room:message`

`room:delegate` is internal-only.
It is a tool-loop mechanism, not a public syscall clients should rely on.

### 3. Frame-first core, typed adapters above it

The library core should stay frame-native.
Do not bake in WebSocket, TUI, HTTP, session, or app-runtime concerns.

However, client projects may want typed adapters layered above frames.
That idea is valid, but it should live in separate crates or optional modules,
not in the frame core.

### 4. Same-room ordering matters more than global simplicity

The current `Arc<Mutex<HashMap<String, Room>>>` was acceptable for bootstrap,
but it does not clearly define same-room concurrency semantics.

The target model is:

- per-room worker owns one room's mutable state
- different rooms run concurrently
- same-room mutation is serialized
- cancellation can interrupt in-flight actor tasks

This follows the stronger `prior` room-worker model.

### 5. No synthetic user chrono messages

The original chrono-marker design created invalid or misleading message ordering
once tool calls entered the conversation.

This crate should not inject synthetic `"user"` messages into LLM history.

If time awareness is needed later, it must be expressed in a way that does not
break assistant/tool adjacency.

### 6. Streaming is a hard requirement

Provider clients must continue to stream incrementally.
The library should not regress into buffered full-response handling.

Streaming expectations apply at three layers:

- provider wire stream
- `llm:chat` item frames
- `room:message` reply delta frames

---

## Core Contracts

## `llm:chat`

Purpose:
- raw provider-backed chat call with history, context, and optional tools

Request data:
- `config: String`
- `history: Vec<HistoryEntry>`
- `context: Vec<Message>` optional
- `tools: Vec<Tool>` optional

Response stream:
- `item { type: "text_delta", text }`
- `item { type: "thinking_delta", thinking }`
- `item { type: "tool_use_delta", index, id, name, input }`
- `item { type: "done", stop_reason, model, input_tokens, output_tokens }`
- `done`

Notes:
- no tool loop here
- no room mutation here
- provider-specific payload construction stays inside the LLM subsystem

## `room:message`

Purpose:
- full room ReAct entry point

Request data:
- `room: String`
- `from: String`
- `content: String`
- `tool_prefixes: Vec<String>` optional
- `tools: Vec<Tool>` optional

Response stream:
- `item { type: "text_delta", actor, room, text }`
- `item { type: "reply", actor, room, content }`
- `done`

Side-channel broadcasts:
- `door:thought`
- `door:tool`
- `door:chat`

Notes:
- actors in a room may run in parallel for a single message
- room history append semantics must remain deterministic

## `room:join`

Request data:
- `room: String`
- `actor_name: String`
- `config: String`

Response:
- `done`

## `room:part`

Request data:
- `room: String`
- `actor_name: String`

Response:
- `done`

## `room:history`

Request data:
- `room: String`
- `limit: u64` optional

Response:
- `item { id, ts, from, content, kind }*`
- `done`

## `room:list`

Request data:
- `{}`

Response:
- `item { room }*`
- `done`

---

## Event Contracts

These broadcasts are emitted as request-style frames for subscribers.

## `door:thought`

Payload:
- `room`
- `from`
- `content`

Semantics:
- emitted for meaningful model thinking/thought output
- best-effort only

## `door:tool`

Payload:
- `room`
- `from`
- `syscall`
- `args` optional

Semantics:
- emitted when an actor initiates a tool call
- describes intent, not result

## `door:chat`

Payload:
- `room`
- `from`
- `content`

Semantics:
- emitted when an actor finishes a visible reply
- informational broadcast for client integrations

---

## Architecture Targets

### LLM subsystem

Keep:

- provider-neutral `ContentBlock`, `ContentDelta`, `Message`, `Tool`
- incremental stream parsing
- config-driven provider selection
- prompt bundle assembly

Add or improve:

- explicit provider fixture tests
- stricter parsing guarantees for multi-tool and mixed-block streams
- optional future support for additional OpenAI API modes only if justified

Do not add yet:

- `llm:models`
- provider-specific public syscalls

### Room subsystem

Current state:

- per-room worker architecture is now in place
- direct top-level routing to room worker mailboxes
- same-room mutation is serialized by the worker
- cancellation is threaded into in-flight `room:message` actor work
- room state remains in-memory only

Target state:

- top-level room router
- one worker task per active room
- serialized same-room mutation
- cancellation-aware actor task management
- explicit in-flight message policy for same room

### Cross-turn tool outcomes

Current state:

- tool calls and raw tool results are private to an actor's current turn
- only the final assistant reply is persisted into shared room history
- later turns can see the reply text, but not the structured tool provenance

Observed limitation:

- later turns do not know which tools were attempted
- later turns do not know whether a prior tool failed or succeeded
- the model can repeat failed tool attempts because prior failure context is
  dropped at turn boundaries

Recommended direction:

- do not persist full raw tool transcripts in normal room history
- add a separate persisted room evidence/outcome lane for compact tool results
- preserve at least:
  - actor
  - syscall/tool name
  - success/failure
  - short summary
  - error code/class when available
  - timestamp and turn association

Intended use:

- inject selected recent tool outcomes into later turns when relevant
- give the model failure/success provenance without replaying large payloads
- reduce pointless retried failing tool calls across turns

Non-goal for this feature:

- storing every raw tool payload forever as conversational history

### Door/event layer

This crate should emit room activity events, but should not become a gateway or
session manager itself.

Meaning:

- emit `door:*` frames
- do not own WebSocket/HTTP/IRC transport concerns
- do not import Abbot-specific session abstractions

---

## Testing Strategy

Testing needs to move closer to `prior`'s pipe-level coverage.

### Required unit coverage

- provider stream parsing
- tool-use block reconstruction
- prompt bundle assembly
- config validation
- history/message conversion

### Required pipe-level integration coverage

- `room:message` text-only path
- tool-use follow-up round
- multiple tool calls in one round
- nested delegation path
- parallel actor replies
- room history persistence across messages
- cancellation during in-flight actor work
- `door:*` broadcast emission
- same-room busy/reject behavior until worker serialization lands

### Required concurrency coverage

- concurrent messages in different rooms overlap correctly
- concurrent messages in the same room follow the chosen policy

### Required regression coverage

- no synthetic user turn inserted between assistant tool call and tool result
- OpenAI multi-tool chunks are preserved
- Anthropic indexed tool fragments remain separated
- streamed usage accounting is preserved

---

## Next Phases

## Phase 1 — Freeze contracts

- document the exact frame shapes listed above
- mark `room:delegate` internal-only in docs and code comments
- ensure `door:*` semantics are documented as library events

Exit criteria:
- public contracts are written down and match the implementation

Status:
- complete

## Phase 2 — Restore `door:*` emissions

- add `door:thought`
- add `door:tool`
- add `door:chat`
- make them best-effort, fire-and-forget
- add tests proving they emit during tool/thought/final reply flows

Exit criteria:
- direct request/response behavior unchanged
- observers can subscribe to room activity events

Status:
- complete

## Phase 3 — Room worker refactor

- replace global mutex room state with per-room workers
- define same-room in-flight policy
  - recommended initial rule: one `room:message` at a time per room
- make cancels reach actor tasks predictably

Exit criteria:
- same-room ordering semantics are explicit
- multi-room concurrency is improved without history races

Status:
- not complete
- current stopgap: same-room overlap is rejected by an in-flight guard
- next implementation target: replace the guard with per-room workers

Refined checklist:
- define worker ownership and lifecycle for room creation/removal
- move room mutation behind worker message passing
- preserve `room:list`, `room:history`, and membership operations cleanly
- define whether same-room requests queue or reject while one is active
- thread cancellation into actor tasks and worker shutdown semantics

## Phase 4 — Test hardening

- port the strongest `prior` room/message tests
- add streamed-provider fixture coverage
- add cancellation and broadcast tests

Exit criteria:
- core ReAct paths and event emissions are covered end-to-end

Status:
- in progress
- regression coverage now exists for:
  - assistant/tool adjacency preservation
  - OpenAI multi-tool chunks
  - Anthropic indexed tool reconstruction
  - malformed Anthropic tool JSON
  - premature `llm:chat` stream termination handling
  - same-room overlap guard behavior

Remaining high-value gaps:
- full pipe-level `room:message` integration tests
- cancellation propagation tests across room and llm boundaries
- multi-room concurrency tests
- broadcast coverage in full message flows, not only helper-level tests

## Phase 5 — Optional adapter layer

- evaluate a separate typed adapter for client crates that want a higher-level
  interface over `door:*` and `room:*`
- keep the frame core separate from those helpers

Exit criteria:
- client ergonomics improve without polluting the frame-native core

---

## Non-Goals For Now

- transport/gateway implementation
- session management
- persistent room membership model
- Abbot-specific external tool wait orchestration
- speculative new syscalls without concrete client need

---

## Immediate Recommendation

The next implementation step should be:

1. refactor room ownership to per-room workers
2. then expand pipe-level and cancellation integration coverage

That sequence replaces the temporary same-room guard with the intended
architecture, then hardens the new execution model with stronger integration
tests.
