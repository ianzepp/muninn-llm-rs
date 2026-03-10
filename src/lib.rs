//! muninn-llm — LLM tool/context/ReAct/room loop for the Muninn kernel family.
//!
//! OVERVIEW
//! ========
//! Provides two `Syscall` implementations for `muninn-kernel`:
//!
//! - [`LlmSyscall`] — handles `llm:chat` requests, streaming provider deltas
//!   as `Item` frames. Supports Anthropic and OpenAI.
//!
//! - [`RoomSyscall`] — handles `room:*` requests. Maintains in-memory room
//!   state (actors, history) and runs the ReAct tool loop for `room:message`.
//!
//! USAGE
//! =====
//! Parse a config, create the two syscall handlers, and register them
//! with the muninn-kernel builder. See `LlmSyscall` and `RoomSyscall`
//! for details.

pub mod config;
pub mod error;
pub mod prompt_bundle;
pub mod types;

pub mod room;

mod client_anthropic;
mod client_openai;
mod llm;

pub use llm::LlmSyscall;
pub use room::RoomSyscall;
