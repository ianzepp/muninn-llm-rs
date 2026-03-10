//! LlmSyscall — `llm:` prefix handler for the Muninn kernel.
//!
//! ARCHITECTURE
//! ============
//! `LlmSyscall` implements `muninn_kernel::Syscall` for the `"llm"` prefix.
//! Each `llm:chat` request is dispatched into a spawned task so concurrent
//! rooms make overlapping API calls without head-of-line blocking.
//!
//! VERBS
//! =====
//! - `llm:chat` — full streaming chat with history, context, and optional tools.
//!
//! CONSTRUCTION
//! ============
//! Create via `LlmSyscall::new(config)` and register with the kernel builder.

pub(crate) mod chat;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::warn;
use uuid::Uuid;

use muninn_kernel::frame::{ErrorCode, Frame, Status};
use muninn_kernel::pipe::Caller;
use muninn_kernel::sender::FrameSender;
use muninn_kernel::syscall::Syscall;

use crate::config::ConfigFile;
use crate::error::LlmError;

use chat::{ProviderClient, build_clients, handle_chat};

// =============================================================================
// CONSTRUCTION
// =============================================================================

/// Kernel syscall handler for the `"llm"` prefix.
pub struct LlmSyscall {
    config: Arc<ConfigFile>,
    clients: Arc<HashMap<String, ProviderClient>>,
}

impl LlmSyscall {
    /// Create a new `LlmSyscall`. Builds all provider clients eagerly so
    /// missing API keys fail at construction time, not on first request.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError`] if any profile has a missing or invalid API key,
    /// or if the HTTP client cannot be constructed.
    pub fn new(config: ConfigFile) -> Result<Self, LlmError> {
        let clients = build_clients(&config)?;
        Ok(Self { config: Arc::new(config), clients: Arc::new(clients) })
    }
}

// =============================================================================
// SYSCALL IMPL
// =============================================================================

#[async_trait]
impl Syscall for LlmSyscall {
    fn prefix(&self) -> &'static str {
        "llm"
    }

    async fn dispatch(
        &self,
        frame: &Frame,
        tx: &FrameSender,
        _caller: &Caller,
        cancel: CancellationToken,
    ) -> Result<(), Box<dyn ErrorCode + Send>> {
        match frame.verb() {
            "chat" => {
                let frame = frame.clone();
                let config = Arc::clone(&self.config);
                let clients = Arc::clone(&self.clients);
                let tx_clone = tx.clone();
                // Spawn so the kernel dispatch loop is not blocked by HTTP.
                // Cancellation is best-effort: the spawned task checks nothing
                // itself, but dropping the sender will terminate stream consumers.
                let _cancel = cancel;
                tokio::spawn(async move {
                    handle_chat(frame, config, clients, &tx_clone).await;
                });
                // The spawned task sends Done/Error; we return Ok here so the
                // kernel does not double-send an error frame.
                Ok(())
            }
            other => Err(Box::new(LlmError::UnknownConfig { name: format!("unknown llm verb: {other}") })),
        }
    }
}
