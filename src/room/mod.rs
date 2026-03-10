//! RoomSyscall — `room:` prefix handler for the Muninn kernel.
//!
//! ARCHITECTURE
//! ============
//! `RoomSyscall` routes requests to one worker task per room. Each worker owns
//! that room's mutable state (actors + history) and processes requests
//! sequentially from its mailbox. Different rooms can progress concurrently;
//! same-room operations are serialized by the worker.
//!
//! VERBS
//! =====
//! - `room:join`    — add an actor to a room (creates the room if absent)
//! - `room:part`    — remove an actor from a room (deletes room when empty)
//! - `room:history` — stream history entries for a room
//! - `room:list`    — stream active room names
//! - `room:message` — user message → streaming ReAct loop → reply items

mod message;
pub mod state;
mod worker;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use muninn_kernel::frame::{ErrorCode, Frame};
use muninn_kernel::pipe::Caller;
use muninn_kernel::sender::FrameSender;
use muninn_kernel::syscall::Syscall;

use crate::config::ConfigFile;
use crate::error::RoomError;
use crate::types::{Data, Tool};

use worker::{WorkerRequest, start_worker};

const WORKER_CAPACITY: usize = 32;

type WorkerMap = HashMap<String, mpsc::Sender<WorkerRequest>>;

/// Kernel syscall handler for the `"room"` prefix.
///
/// Top-level state tracks room worker mailboxes only. Each worker owns the
/// mutable state for a single room.
///
/// WHY `Arc<Mutex<WorkerMap>>`: `Syscall::dispatch` takes `&self` so all
/// routing state must be interior-mutable. The lock is held only briefly
/// (map lookup or insert), never across await points.
///
/// WHY `Arc<Mutex<UnboundedReceiver>>`: the `Syscall` trait requires `Sync`.
/// `UnboundedReceiver` is `!Sync`, so it must be wrapped in a `Mutex`.
/// Only one thread calls `dispatch` at a time in practice (the kernel
/// dispatch loop), so the mutex is uncontended.
pub struct RoomSyscall {
    workers: Arc<Mutex<WorkerMap>>,
    /// Sender given to each worker so it can signal when its room is empty.
    cleanup_tx: mpsc::UnboundedSender<String>,
    /// Receiver drained at the start of each dispatch call to remove stale entries.
    cleanup_rx: Arc<Mutex<mpsc::UnboundedReceiver<String>>>,
}

impl RoomSyscall {
    /// Create a new `RoomSyscall`. The config is accepted for API consistency
    /// but is not currently used — room state is ephemeral and in-memory only.
    #[must_use]
    pub fn new(_config: ConfigFile) -> Self {
        let (cleanup_tx, cleanup_rx) = mpsc::unbounded_channel();
        Self {
            workers: Arc::new(Mutex::new(HashMap::new())),
            cleanup_tx,
            cleanup_rx: Arc::new(Mutex::new(cleanup_rx)),
        }
    }

    /// Remove stale worker mailbox entries for rooms that have shut down.
    ///
    /// WHY unbounded channel + drain-on-dispatch: workers signal cleanup by
    /// sending their room name on `cleanup_tx` when the last actor parts.
    /// Rather than acquiring the workers lock from the worker task (which
    /// would create a potential deadlock), the cleanup messages are drained
    /// here, at the start of each `dispatch` call, under a short-lived lock.
    /// This amortizes cleanup across requests without background tasks.
    fn drain_cleanup(&self) {
        let mut cleanup = self
            .cleanup_rx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        while let Ok(room) = cleanup.try_recv() {
            workers.remove(&room);
        }
    }

    /// Extract the required `"room"` field from frame data.
    fn room_from_frame(frame: &Frame) -> Result<String, RoomError> {
        frame
            .data
            .get("room")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| RoomError::Deserialize("missing 'room' field".into()))
    }

    /// Return the sender for an existing room worker, or create a new one.
    ///
    /// When `create` is `false` and no worker exists, returns
    /// [`RoomError::RoomNotFound`] instead of spawning a new worker. This
    /// distinguishes operations that must target an existing room (`room:part`,
    /// `room:history`) from those that create rooms implicitly (`room:join`,
    /// `room:message`).
    fn worker_sender(
        &self,
        room: &str,
        create: bool,
    ) -> Result<mpsc::Sender<WorkerRequest>, RoomError> {
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if let Some(sender) = workers.get(room) {
            return Ok(sender.clone());
        }

        if !create {
            return Err(RoomError::RoomNotFound {
                room: room.to_string(),
            });
        }

        let (tx, rx) = mpsc::channel(WORKER_CAPACITY);
        start_worker(room.to_string(), rx, self.cleanup_tx.clone());
        workers.insert(room.to_string(), tx.clone());
        Ok(tx)
    }

    /// Send a request to the named room's worker, with one automatic retry on failure.
    ///
    /// WHY retry on send failure: the channel send can fail if the worker task
    /// exited between `worker_sender` returning and the `send` completing. This
    /// is a race condition between the cleanup drain and a concurrent request.
    /// On failure we remove the stale entry and, if `create` is set, spawn a
    /// fresh worker and retry once. A second failure is treated as terminal.
    async fn dispatch_to_worker(
        &self,
        room: &str,
        create: bool,
        frame: &Frame,
        tx: &FrameSender,
        caller: &Caller,
        cancel: CancellationToken,
    ) -> Result<(), Box<dyn ErrorCode + Send>> {
        let sender = self.worker_sender(room, create).map_err(box_err)?;
        let request = WorkerRequest {
            frame: frame.clone(),
            tx: tx.clone(),
            caller: caller.clone(),
            cancel: cancel.clone(),
        };

        if sender.send(request).await.is_err() {
            // WHY remove before re-acquiring sender: we cannot call worker_sender
            // while holding the stale sender, since worker_sender also locks workers.
            {
                let mut workers = self
                    .workers
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                workers.remove(room);
            }

            if !create {
                return Err(box_err(RoomError::RoomNotFound {
                    room: room.to_string(),
                }));
            }

            let sender = self.worker_sender(room, true).map_err(box_err)?;
            sender
                .send(WorkerRequest {
                    frame: frame.clone(),
                    tx: tx.clone(),
                    caller: caller.clone(),
                    cancel,
                })
                .await
                .map_err(|_| box_msg(format!("room worker closed while handling room '{room}'")))?;
        }

        Ok(())
    }
}

#[async_trait]
impl Syscall for RoomSyscall {
    fn prefix(&self) -> &'static str {
        "room"
    }

    async fn dispatch(
        &self,
        frame: &Frame,
        tx: &FrameSender,
        caller: &Caller,
        cancel: CancellationToken,
    ) -> Result<(), Box<dyn ErrorCode + Send>> {
        self.drain_cleanup();

        match frame.verb() {
            "list" => {
                let rooms: Vec<String> = {
                    let workers = self
                        .workers
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    workers.keys().cloned().collect()
                };

                for room in rooms {
                    let mut data = Data::new();
                    data.insert("room".into(), room.into());
                    tx.send(frame.item(data))
                        .await
                        .map_err(|e| box_msg(e.to_string()))?;
                }
                tx.send_done(frame)
                    .await
                    .map_err(|e| box_msg(e.to_string()))?;
                Ok(())
            }
            "join" | "message" => {
                let room = Self::room_from_frame(frame).map_err(box_err)?;
                self.dispatch_to_worker(&room, true, frame, tx, caller, cancel)
                    .await
            }
            "part" | "history" => {
                let room = Self::room_from_frame(frame).map_err(box_err)?;
                self.dispatch_to_worker(&room, false, frame, tx, caller, cancel)
                    .await
            }
            other => Err(box_msg(format!("unknown room verb: {other}"))),
        }
    }
}

/// Deserialize the optional `"tools"` field from a `Data` map.
///
/// Returns an empty `Vec` if the field is absent — tools are optional for
/// all `room:*` verbs. Returns a deserialization error only if the field is
/// present but malformed.
pub(crate) fn parse_tools_field(data: &Data) -> Result<Vec<Tool>, RoomError> {
    let Some(value) = data.get("tools") else {
        return Ok(Vec::new());
    };

    serde_json::from_value(value.clone())
        .map_err(|e| RoomError::Deserialize(format!("invalid 'tools' field: {e}")))
}

/// Extract a required string field from a `Data` map.
pub(crate) fn str_field(data: &Data, key: &str) -> Result<String, RoomError> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RoomError::Deserialize(format!("missing '{key}' field")))
}

/// Box an [`ErrorCode`] implementor for return from `Syscall::dispatch`.
fn box_err(e: impl ErrorCode + Send + 'static) -> Box<dyn ErrorCode + Send> {
    Box::new(e)
}

/// Construct a boxed [`LlmError::InternalCall`] from a plain message string.
fn box_msg(msg: String) -> Box<dyn ErrorCode + Send> {
    Box::new(crate::error::LlmError::InternalCall(msg))
}

#[cfg(test)]
#[path = "mod_test.rs"]
mod tests;
