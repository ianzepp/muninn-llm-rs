//! muninn-llm — LLM tool/context/ReAct/room loop for the Muninn kernel family.
//!
//! SYSTEM CONTEXT
//! ==============
//! This crate sits between `muninn-kernel` (the frame-routing bus) and external
//! LLM providers (Anthropic, OpenAI). The kernel dispatches `Frame` values to
//! registered `Syscall` handlers; this crate provides two such handlers.
//!
//! OVERVIEW
//! ========
//! Provides two `Syscall` implementations for `muninn-kernel`:
//!
//! - [`LlmSyscall`] — handles `llm:chat` requests, streaming provider deltas
//!   as `Item` frames. Supports Anthropic and OpenAI. Each request is serviced
//!   by a spawned task so concurrent rooms never block each other.
//!
//! - [`RoomSyscall`] — handles `room:*` requests. Maintains in-memory room
//!   state (actors, history) and runs the ReAct tool loop for `room:message`.
//!   One worker task per room serializes mutations while allowing rooms to
//!   progress concurrently.
//!
//! USAGE
//! =====
//! Parse a config, create the two syscall handlers, and register them
//! with the muninn-kernel builder. See [`LlmSyscall`] and [`RoomSyscall`]
//! for construction details.
//!
//! ```ignore
//! let config = muninn_llm::config::load_config(path)?;
//! let llm = LlmSyscall::new(config.clone())?;
//! let room = RoomSyscall::new(config);
//! // register with kernel builder ...
//! ```

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
