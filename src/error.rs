//! Error type for the muninn-llm subsystem.

use muninn_kernel::frame::ErrorCode;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("config load failed: {0}")]
    ConfigLoad(String),

    #[error("config parse failed: {0}")]
    ConfigParse(String),

    #[error("missing API key: env var {var} not set")]
    MissingApiKey { var: String },

    #[error("unknown config: {name}")]
    UnknownConfig { name: String },

    #[error("API request failed: {0}")]
    ApiRequest(String),

    #[error("API response error: status {status}")]
    ApiResponse { status: u16, body: String },

    #[error("API response parse failed: {0}")]
    ApiParse(String),

    #[error("stream decode failed: {0}")]
    StreamDecode(String),

    #[error("deserialization failed: {0}")]
    Deserialize(String),

    #[error("serialization failed: {0}")]
    Serialize(String),

    #[error("internal call failed: {0}")]
    InternalCall(String),

    #[error("HTTP client build failed: {0}")]
    HttpClientBuild(String),

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

#[derive(Debug, thiserror::Error)]
pub enum RoomError {
    #[error("room not found: {room}")]
    RoomNotFound { room: String },

    #[error("room busy: {room}")]
    RoomBusy { room: String },

    #[error("actor not found: {name} in room {room}")]
    ActorNotFound { room: String, name: String },

    #[error("actor already joined: {name} in room {room}")]
    ActorAlreadyJoined { room: String, name: String },

    #[error("deserialization failed: {0}")]
    Deserialize(String),

    #[error("serialization failed: {0}")]
    Serialize(String),

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
