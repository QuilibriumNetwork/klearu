//! Unified error type for desktop commands. All commands return `Result<T,
//! DesktopError>` and Tauri serializes the error to a string for the JS
//! side; the `Serialize` impl emits a `{ code, message }` object so the UI
//! can branch on `code` instead of regex-matching strings.

use serde::{Serialize, Serializer};

#[derive(Debug, thiserror::Error)]
pub enum DesktopError {
    #[error("not loaded: {0}")]
    NotLoaded(&'static str),
    #[error("bad input: {0}")]
    BadInput(String),
    #[error("storage: {0}")]
    Storage(String),
    #[error("engine: {0}")]
    Engine(String),
    #[error("io: {0}")]
    Io(String),
    #[error("aborted")]
    Aborted,
    #[error("{0}")]
    Other(String),
}

impl DesktopError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotLoaded(_) => "not_loaded",
            Self::BadInput(_) => "bad_input",
            Self::Storage(_) => "storage",
            Self::Engine(_) => "engine",
            Self::Io(_) => "io",
            Self::Aborted => "aborted",
            Self::Other(_) => "other",
        }
    }
    pub fn bad_input(s: impl Into<String>) -> Self { Self::BadInput(s.into()) }
    pub fn storage(s: impl Into<String>) -> Self { Self::Storage(s.into()) }
    pub fn engine(s: impl Into<String>) -> Self { Self::Engine(s.into()) }
    pub fn from_rusqlite(e: rusqlite::Error) -> Self { Self::Storage(e.to_string()) }
}

impl From<std::io::Error> for DesktopError {
    fn from(e: std::io::Error) -> Self { Self::Io(e.to_string()) }
}

impl From<klearu_llm::LlmError> for DesktopError {
    fn from(e: klearu_llm::LlmError) -> Self { Self::Engine(e.to_string()) }
}

impl Serialize for DesktopError {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("DesktopError", 2)?;
        st.serialize_field("code", self.code())?;
        st.serialize_field("message", &self.to_string())?;
        st.end()
    }
}

pub type Result<T> = std::result::Result<T, DesktopError>;
