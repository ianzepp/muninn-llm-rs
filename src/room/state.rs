//! Room state — actors, history entries, and the in-memory Room struct.
//!
//! DESIGN
//! ======
//! Room state lives entirely in the `RoomSyscall` task — no shared memory,
//! no mutex. All mutation goes through the single-task receive loop.
//!
//! WHY `id` + `ts`: `id` is a monotonic counter for stable ordering and
//! pagination. `ts` is wall-clock seconds for display. Relying on `ts` alone
//! breaks if entries arrive in the same second or out of order.

use serde::{Deserialize, Serialize};

use crate::error::RoomError;

// =============================================================================
// ACTOR
// =============================================================================

/// An LLM actor assigned to a room.
///
/// `config` is an opaque string (e.g., "default", "hand") that references
/// a named profile in `ConfigFile`. The room subsystem never interprets it —
/// it passes it through to `llm:chat` requests.
#[derive(Debug, Clone)]
pub struct Actor {
    pub name: String,
    pub config: String,
}

// =============================================================================
// HISTORY
// =============================================================================

/// The role of a history entry — maps to LLM message roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HistoryKind {
    User,
    Assistant,
}

impl HistoryKind {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A single entry in a room's conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: u64,
    /// Unix seconds.
    pub ts: i64,
    pub from: String,
    pub content: String,
    pub kind: HistoryKind,
}

/// A compact persisted summary of a tool outcome from a completed turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutcomeRecord {
    pub actor: String,
    pub syscall: String,
    pub ok: bool,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    pub ts: i64,
    pub turn_id: u64,
}

// =============================================================================
// ROOM
// =============================================================================

/// In-memory state for a single room.
pub struct Room {
    pub actors: Vec<Actor>,
    pub history: Vec<HistoryEntry>,
    pub tool_outcomes: Vec<ToolOutcomeRecord>,
    next_id: u64,
}

impl Room {
    #[must_use]
    pub fn new() -> Self {
        Self {
            actors: Vec::new(),
            history: Vec::new(),
            tool_outcomes: Vec::new(),
            next_id: 1,
        }
    }

    /// Append a message to history and return its assigned id.
    ///
    /// # Errors
    ///
    /// Returns [`RoomError::IdOverflow`] if the monotonic counter overflows.
    pub fn add_message(
        &mut self,
        from: impl Into<String>,
        content: impl Into<String>,
        kind: HistoryKind,
    ) -> Result<u64, RoomError> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(RoomError::IdOverflow)?;
        let ts = now_secs();
        self.history.push(HistoryEntry {
            id,
            ts,
            from: from.into(),
            content: content.into(),
            kind,
        });
        Ok(id)
    }

    /// Remove oldest entries so history length stays at or below `max`.
    pub fn trim_history(&mut self, max: usize) {
        if self.history.len() > max {
            let remove = self.history.len() - max;
            self.history.drain(..remove);
        }
    }

    pub fn add_tool_outcome(&mut self, outcome: ToolOutcomeRecord) {
        self.tool_outcomes.push(outcome);
    }

    #[must_use]
    pub fn render_recent_tool_outcomes(&self, limit: usize) -> String {
        if limit == 0 || self.tool_outcomes.is_empty() {
            return String::new();
        }

        self.tool_outcomes
            .iter()
            .rev()
            .take(limit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|outcome| {
                let status = if outcome.ok { "ok" } else { "error" };
                let mut line = format!(
                    "- turn {} | actor {} | {} | {} | {}",
                    outcome.turn_id, outcome.actor, outcome.syscall, status, outcome.summary
                );
                if let Some(code) = &outcome.error_code {
                    line.push_str(" (code: ");
                    line.push_str(code);
                    line.push(')');
                }
                line
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for Room {
    fn default() -> Self {
        Self::new()
    }
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

#[cfg(test)]
#[path = "state_test.rs"]
mod tests;
