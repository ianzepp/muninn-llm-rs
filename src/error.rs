//! Error types for the muninn-llm subsystem.
//!
//! DESIGN
//! ======
//! Two separate error enums exist because the two syscall surfaces have
//! non-overlapping failure modes. [`LlmError`] covers provider communication
//! and config loading — concerns owned by `LlmSyscall` and the client layer.
//! [`RoomError`] covers room and actor lifecycle — concerns owned by
//! `RoomSyscall` and its per-room workers.
//!
//! Both implement `muninn_kernel::frame::ErrorCode`, which the kernel uses to
//! attach a machine-readable `code` field to error frames sent upstream.
//!
//! ERROR HANDLING STRATEGY
//! =======================
//! - Provider errors (HTTP 4xx/5xx) surface as `LlmError::ApiResponse`.
//! - Network errors (connection reset, timeout) surface as `LlmError::ApiRequest`.
//! - The `retryable()` impl on `LlmError` flags errors the caller may safely
//!   retry: transient network failures and rate-limit / server-error responses.
//! - All other errors are terminal for the current request.

use muninn_kernel::frame::ErrorCode;

/// Errors produced by the LLM provider layer and config subsystem.
///
/// Implements [`muninn_kernel::frame::ErrorCode`] so the kernel can attach
/// a stable `code` string to error frames sent upstream to callers.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The config file could not be read from disk.
    #[error("config load failed: {0}")]
    ConfigLoad(String),

    /// The config file was read but failed TOML parsing or schema validation.
    #[error("config parse failed: {0}")]
    ConfigParse(String),

    /// The profile requires an API key but neither `api_key_value` nor the
    /// named env var (`api_key_env`) is set.
    #[error("missing API key: env var {var} not set")]
    MissingApiKey { var: String },

    /// A `config` name in a request does not match any profile in `ConfigFile`.
    #[error("unknown config: {name}")]
    UnknownConfig { name: String },

    /// The HTTP request to the provider could not be sent (network error,
    /// connection refused, timeout before first byte, etc.).
    #[error("API request failed: {0}")]
    ApiRequest(String),

    /// The provider returned a non-200 HTTP status. `body` contains the raw
    /// response body for diagnosis.
    #[error("API response error: status {status}")]
    ApiResponse { status: u16, body: String },

    /// The provider returned 200 but the response body could not be parsed.
    #[error("API response parse failed: {0}")]
    ApiParse(String),

    /// An SSE or NDJSON chunk from the provider stream could not be decoded.
    #[error("stream decode failed: {0}")]
    StreamDecode(String),

    /// A frame `data` map could not be deserialized into the expected type.
    #[error("deserialization failed: {0}")]
    Deserialize(String),

    /// A typed value could not be serialized into a frame `data` map.
    #[error("serialization failed: {0}")]
    Serialize(String),

    /// An internal `llm:chat` or `room:delegate` call through the kernel
    /// pipe failed or returned an error frame.
    #[error("internal call failed: {0}")]
    InternalCall(String),

    /// The `reqwest` HTTP client could not be constructed (invalid TLS config, etc.).
    #[error("HTTP client build failed: {0}")]
    HttpClientBuild(String),

    /// Sending a frame through a `FrameSender` channel failed because the
    /// receiver was dropped.
    #[error("pipe send failed: {0}")]
    PipeSend(String),
}

impl ErrorCode for LlmError {
    fn error_code(&self) -> &'static str {
        match self {
            Self::ConfigLoad(_) => "E_CONFIG_LOAD",
            Self::ConfigParse(_) => "E_CONFIG_PARSE",
            Self::MissingApiKey { .. } => "E_MISSING_API_KEY",
            Self::UnknownConfig { .. } => "E_UNKNOWN_CONFIG",
            Self::ApiRequest(_) => "E_API_REQUEST",
            Self::ApiResponse { .. } => "E_API_RESPONSE",
            Self::ApiParse(_) => "E_API_PARSE",
            Self::StreamDecode(_) => "E_STREAM_DECODE",
            Self::Deserialize(_) => "E_DESERIALIZE",
            Self::Serialize(_) => "E_SERIALIZE",
            Self::InternalCall(_) => "E_INTERNAL_CALL",
            Self::HttpClientBuild(_) => "E_HTTP_CLIENT_BUILD",
            Self::PipeSend(_) => "E_PIPE_SEND",
        }
    }

    /// Returns `true` for errors a caller may safely retry without changes.
    ///
    /// `ApiRequest` covers network-level transients (connection reset, timeout).
    /// `ApiResponse` with 429 or 5xx covers rate-limiting and transient server
    /// errors where the provider may succeed on a subsequent attempt.
    fn retryable(&self) -> bool {
        matches!(
            self,
            Self::ApiRequest(_)
                | Self::ApiResponse {
                    status: 429 | 500..=599,
                    ..
                }
        )
    }
}

/// Errors produced by the room lifecycle and state management layer.
///
/// These errors arise from invalid caller requests (missing room, duplicate
/// actor) or from internal state integrity failures (id overflow). They are
/// never retryable — each requires a corrected request or system restart.
#[derive(Debug, thiserror::Error)]
pub enum RoomError {
    /// The request targeted a room name that has no active worker. Rooms are
    /// created implicitly by `room:join` and destroyed when the last actor parts.
    #[error("room not found: {room}")]
    RoomNotFound { room: String },

    /// Reserved for future use; not currently emitted.
    #[error("room busy: {room}")]
    RoomBusy { room: String },

    /// A `room:part` or similar request named an actor not present in the room.
    #[error("actor not found: {name} in room {room}")]
    ActorNotFound { room: String, name: String },

    /// A `room:join` request named an actor already present in the room.
    /// Actors must `room:part` before re-joining.
    #[error("actor already joined: {name} in room {room}")]
    ActorAlreadyJoined { room: String, name: String },

    /// A frame `data` map could not be deserialized into the expected type.
    #[error("deserialization failed: {0}")]
    Deserialize(String),

    /// A typed value could not be serialized into a frame `data` map.
    #[error("serialization failed: {0}")]
    Serialize(String),

    /// The monotonic history entry counter has exhausted `u64`. This is
    /// effectively unreachable in practice but is checked to avoid silent wrap.
    #[error("history id overflow")]
    IdOverflow,
}

impl ErrorCode for RoomError {
    fn error_code(&self) -> &'static str {
        match self {
            Self::RoomNotFound { .. } => "E_ROOM_NOT_FOUND",
            Self::RoomBusy { .. } => "E_ROOM_BUSY",
            Self::ActorNotFound { .. } => "E_ACTOR_NOT_FOUND",
            Self::ActorAlreadyJoined { .. } => "E_ACTOR_ALREADY_JOINED",
            Self::Deserialize(_) => "E_DESERIALIZE",
            Self::Serialize(_) => "E_SERIALIZE",
            Self::IdOverflow => "E_ID_OVERFLOW",
        }
    }
}
