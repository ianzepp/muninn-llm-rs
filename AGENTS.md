# AGENTS.md

## Purpose

`muninn-llm-rs` is a Rust library for projects that already use a
message/frame/event architecture and need an embedded LLM loop.

This crate is:
- frame-native
- transport-agnostic
- streaming-first
- intended to emit both direct response frames and side-channel `door:*` events

When making changes, preserve that shape. Do not turn this crate into an app
runtime, provider-specific wrapper, or transport-specific integration layer.

## Architecture Priorities

Use [PLAN_V2.md](/Users/ianzepp/github/ianzepp/muninn-llm-rs/PLAN_V2.md) as
the current architecture plan.

Important current architectural rules:
- `door:thought`, `door:tool`, and `door:chat` are core library events
- `room:message` is the main ReAct/tool-loop entry point
- provider clients must stream incrementally; do not buffer full responses
- do not inject synthetic `"user"` chrono messages into chat history
- keep the core frame-first; higher-level typed adapters belong above this layer

## Rust Review Standard

Apply the `rust-correctness-surgeon` rubric to nontrivial Rust changes.

The review standard here is correctness-first, not style-first:
- hunt for silent data loss, swallowed errors, invalid ordering, weakened
  invariants, concurrency mistakes, and incorrect defaults
- treat `.ok()`, `let _ =`, `unwrap_or_default()`, and similar error-dropping
  patterns as suspicious and justify them explicitly
- prefer fixing the root cause over adding fallbacks that hide incorrectness

When reviewing or summarizing findings, use this structure:
- severity
- location
- what is wrong
- why it matters with a concrete failure scenario
- specific fix

## Testing Conventions

Match the `muninn-kernel` test layout and hygiene patterns.

Rules:
- prefer sibling `*_test.rs` files over inline `mod tests`
- keep tests focused and small; production code should remain testable without
  deep harness setup
- add or update a dedicated hygiene test when project-wide conventions change
- prefer pipe/frame-level tests for frame contracts and small targeted unit
  tests for parsing/state helpers
- add regression tests for correctness bugs before or alongside fixes

Current verification baseline:
- `cargo fmt`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`

Run all three before considering the task complete unless the user explicitly
asks for something narrower.

## Change Discipline

- preserve streaming semantics across providers and room orchestration
- preserve assistant/tool adjacency in message history and provider payloads
- keep public frame contracts stable unless the task is explicitly a contract
  change
- prefer surgical fixes over broad refactors unless the architectural target is
  clear
- if concurrency semantics are involved, state the invariant being protected
  and add a test for it

## File/Module Guidance

- keep provider-specific payload building and stream decoding inside provider
  client modules
- keep room orchestration concerns inside the room subsystem
- avoid leaking app-specific concerns into the frame core
- add brief comments only where the invariant or ordering rule is not obvious

## Documentation Updates

Update docs when behavior or contracts change:
- [README.md](/Users/ianzepp/github/ianzepp/muninn-llm-rs/README.md) for
  library-facing usage or positioning changes
- [PLAN_V2.md](/Users/ianzepp/github/ianzepp/muninn-llm-rs/PLAN_V2.md) for
  architecture direction changes
- this file for repo workflow or engineering standard changes
