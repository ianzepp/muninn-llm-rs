use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{info, warn};

use muninn_kernel::frame::Frame;
use muninn_kernel::pipe::Caller;
use muninn_kernel::sender::FrameSender;

use crate::error::{LlmError, RoomError};
use crate::room::message;
use crate::room::state::{Actor, HistoryKind, Room, ToolOutcomeRecord};
use crate::types::Data;

use super::{parse_tools_field, str_field};

pub struct WorkerRequest {
    pub frame: Frame,
    pub tx: FrameSender,
    pub caller: Caller,
    pub cancel: tokio_util::sync::CancellationToken,
}

pub fn start_worker(
    room_name: String,
    rx: mpsc::Receiver<WorkerRequest>,
    cleanup_tx: mpsc::UnboundedSender<String>,
) {
    tokio::spawn(async move {
        let mut worker = RoomWorker::new(room_name, cleanup_tx);
        worker.run(rx).await;
    });
}

struct RoomWorker {
    room_name: String,
    room: Room,
    cleanup_tx: mpsc::UnboundedSender<String>,
}

impl RoomWorker {
    fn new(room_name: String, cleanup_tx: mpsc::UnboundedSender<String>) -> Self {
        Self {
            room_name,
            room: Room::default(),
            cleanup_tx,
        }
    }

    async fn run(&mut self, mut rx: mpsc::Receiver<WorkerRequest>) {
        while let Some(request) = rx.recv().await {
            let should_exit = self.handle(request).await;
            if should_exit {
                break;
            }
        }
    }

    async fn handle(&mut self, request: WorkerRequest) -> bool {
        match request.frame.verb() {
            "join" => {
                self.handle_join(request).await;
                false
            }
            "part" => self.handle_part(request).await,
            "history" => {
                self.handle_history(request).await;
                false
            }
            "message" => {
                self.handle_message(request).await;
                false
            }
            other => {
                send_error(
                    &request.tx,
                    &request.frame,
                    format!("unknown room verb: {other}"),
                )
                .await;
                false
            }
        }
    }

    async fn handle_join(&mut self, request: WorkerRequest) {
        let frame = &request.frame;
        let actor_name = match str_field(&frame.data, "actor_name") {
            Ok(value) => value,
            Err(err) => {
                send_error_from(&request.tx, frame, &err).await;
                return;
            }
        };
        let config = match str_field(&frame.data, "config") {
            Ok(value) => value,
            Err(err) => {
                send_error_from(&request.tx, frame, &err).await;
                return;
            }
        };

        if self
            .room
            .actors
            .iter()
            .any(|actor| actor.name == actor_name)
        {
            let err = RoomError::ActorAlreadyJoined {
                room: self.room_name.clone(),
                name: actor_name,
            };
            send_error_from(&request.tx, frame, &err).await;
            return;
        }

        self.room.actors.push(Actor {
            name: actor_name,
            config,
        });
        send_done(&request.tx, frame).await;
    }

    async fn handle_part(&mut self, request: WorkerRequest) -> bool {
        let frame = &request.frame;
        let actor_name = match str_field(&frame.data, "actor_name") {
            Ok(value) => value,
            Err(err) => {
                send_error_from(&request.tx, frame, &err).await;
                return false;
            }
        };

        let before = self.room.actors.len();
        self.room.actors.retain(|actor| actor.name != actor_name);
        if self.room.actors.len() == before {
            let err = RoomError::ActorNotFound {
                room: self.room_name.clone(),
                name: actor_name,
            };
            send_error_from(&request.tx, frame, &err).await;
            return false;
        }

        send_done(&request.tx, frame).await;

        if self.room.actors.is_empty() {
            if let Err(err) = self.cleanup_tx.send(self.room_name.clone()) {
                info!(error = %err, room = %self.room_name, "room: failed to queue worker cleanup");
            }
            return true;
        }

        false
    }

    async fn handle_history(&self, request: WorkerRequest) {
        let frame = &request.frame;
        let limit = frame
            .data
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok());

        let entries: Vec<_> = match limit {
            Some(n) => self
                .room
                .history
                .iter()
                .rev()
                .take(n)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect(),
            None => self.room.history.iter().collect(),
        };

        for entry in entries {
            let mut data = Data::new();
            data.insert("id".into(), entry.id.into());
            data.insert("ts".into(), entry.ts.into());
            data.insert("from".into(), entry.from.clone().into());
            data.insert("content".into(), entry.content.clone().into());
            data.insert("kind".into(), entry.kind.as_str().into());
            if request.tx.send(frame.item(data)).await.is_err() {
                return;
            }
        }

        send_done(&request.tx, frame).await;
    }

    async fn handle_message(&mut self, request: WorkerRequest) {
        let frame = &request.frame;
        let from = match str_field(&frame.data, "from") {
            Ok(value) => value,
            Err(err) => {
                send_error_from(&request.tx, frame, &err).await;
                return;
            }
        };
        let content = match str_field(&frame.data, "content") {
            Ok(value) => value,
            Err(err) => {
                send_error_from(&request.tx, frame, &err).await;
                return;
            }
        };

        let allowed_prefixes: Vec<String> = frame
            .data
            .get("tool_prefixes")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let tools = match parse_tools_field(&frame.data) {
            Ok(value) => value,
            Err(err) => {
                send_error_from(&request.tx, frame, &err).await;
                return;
            }
        };

        if let Err(err) = self.room.add_message(&from, &content, HistoryKind::User) {
            send_error_from(&request.tx, frame, &err).await;
            return;
        }
        let turn_id = self.room.history.last().map_or(0, |entry| entry.id);

        let actors = self.room.actors.clone();
        let history = self.room.history.clone();
        let memory = self.room.render_recent_tool_outcomes(8);

        let mut actor_tasks = JoinSet::new();
        for actor in &actors {
            let actor_name = actor.name.clone();
            let actor_config = actor.config.clone();
            let room_name = self.room_name.clone();
            let history_clone = history.clone();
            let tools_clone = tools.clone();
            let allowed_clone = allowed_prefixes.clone();
            let actors_clone = actors.clone();
            let memory_clone = memory.clone();
            let caller = request.caller.clone();
            let tx = request.tx.clone();
            let frame = request.frame.clone();
            let cancel = request.cancel.clone();

            actor_tasks.spawn(async move {
                tokio::select! {
                    () = cancel.cancelled() => Err(LlmError::InternalCall("room:message cancelled".to_string())),
                    result = message::run_actor_loop(
                        &caller,
                        &actor_config,
                        history_clone,
                        &tools_clone,
                        &allowed_clone,
                        &room_name,
                        &actor_name,
                        &actors_clone,
                        &memory_clone,
                        &tx,
                        &frame,
                    ) => result,
                }
                .map(|result| (actor_name, result))
            });
        }

        let mut cancelled = false;
        while let Some(result) = actor_tasks.join_next().await {
            match result {
                Ok(Ok((actor_name, result))) => {
                    if let Err(err) =
                        self.room
                            .add_message(&actor_name, &result.reply, HistoryKind::Assistant)
                    {
                        warn!(error = %err, actor = %actor_name, room = %self.room_name, "room: failed to persist assistant reply");
                    }
                    for outcome in result.tool_outcomes {
                        self.room.add_tool_outcome(ToolOutcomeRecord {
                            actor: actor_name.clone(),
                            syscall: outcome.syscall,
                            ok: outcome.ok,
                            summary: outcome.summary,
                            error_code: outcome.error_code,
                            ts: self.room.history.last().map_or(0, |entry| entry.ts),
                            turn_id,
                        });
                    }
                }
                Ok(Err(LlmError::InternalCall(msg))) if msg == "room:message cancelled" => {
                    cancelled = true;
                    actor_tasks.abort_all();
                    break;
                }
                Ok(Err(err)) => {
                    warn!(error = %err, room = %self.room_name, "room: actor loop error");
                }
                Err(err) => {
                    warn!(error = %err, room = %self.room_name, "room: actor task join error");
                }
            }
        }

        if cancelled {
            send_error(&request.tx, frame, "room:message cancelled").await;
            return;
        }

        send_done(&request.tx, frame).await;
    }
}

async fn send_done(tx: &FrameSender, frame: &Frame) {
    if let Err(err) = tx.send_done(frame).await {
        info!(error = %err, call = %frame.call, "room: failed to send done frame");
    }
}

async fn send_error(tx: &FrameSender, frame: &Frame, message: impl Into<String>) {
    let message = message.into();
    if let Err(err) = tx.send_error(frame, message.clone()).await {
        info!(error = %err, call = %frame.call, message = %message, "room: failed to send error frame");
    }
}

async fn send_error_from(
    tx: &FrameSender,
    frame: &Frame,
    err_code: &impl muninn_kernel::frame::ErrorCode,
) {
    if let Err(send_err) = tx.send_error_from(frame, err_code).await {
        info!(error = %send_err, call = %frame.call, message = %err_code, "room: failed to send structured error frame");
    }
}
