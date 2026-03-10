# muninn-llm

`muninn-llm` provides the LLM-facing syscalls for the Muninn kernel family.

It currently includes:

- `LlmSyscall` for `llm:chat`
- `RoomSyscall` for in-memory `room:*` coordination and tool loops
- Anthropic and OpenAI chat-completions clients with streamed delta forwarding

The crate is intended to be embedded by a Muninn host process that loads config,
constructs the syscall handlers, and registers them with `muninn-kernel`.
