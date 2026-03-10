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
pub struct RoomSyscall {
    workers: Arc<Mutex<WorkerMap>>,
    cleanup_tx: mpsc::UnboundedSender<String>,
    cleanup_rx: Arc<Mutex<mpsc::UnboundedReceiver<String>>>,
}

impl RoomSyscall {
    #[must_use]
    pub fn new(_config: ConfigFile) -> Self {
        let (cleanup_tx, cleanup_rx) = mpsc::unbounded_channel();
        Self {
            workers: Arc::new(Mutex::new(HashMap::new())),
            cleanup_tx,
            cleanup_rx: Arc::new(Mutex::new(cleanup_rx)),
        }
    }

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

    fn room_from_frame(frame: &Frame) -> Result<String, RoomError> {
        frame
            .data
            .get("room")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| RoomError::Deserialize("missing 'room' field".into()))
    }

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

fn parse_tools_field(data: &Data) -> Result<Vec<Tool>, RoomError> {
    let Some(value) = data.get("tools") else {
        return Ok(Vec::new());
    };

    serde_json::from_value(value.clone())
        .map_err(|e| RoomError::Deserialize(format!("invalid 'tools' field: {e}")))
}

fn str_field(data: &Data, key: &str) -> Result<String, RoomError> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RoomError::Deserialize(format!("missing '{key}' field")))
}

fn box_err(e: impl ErrorCode + Send + 'static) -> Box<dyn ErrorCode + Send> {
    Box::new(e)
}

fn box_msg(msg: String) -> Box<dyn ErrorCode + Send> {
    Box::new(crate::error::LlmError::InternalCall(msg))
}

#[cfg(test)]
#[path = "mod_test.rs"]
mod tests;
