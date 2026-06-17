//! Persistent store for conversations, messages, and assistant memory.
//!
//! Schema mirrors the on-disk auto-memory format (user / feedback / project /
//! reference, with frontmatter name/description/type and a markdown body)
//! so that — later — we can sync the SQLite store with a directory of
//! `*.md` files on disk. The same identifiers are used in both places.
//!
//! All access goes through a single [`MemoryStore`] held behind a mutex
//! inside the Tauri state. Volumes are tiny (conversations + a few hundred
//! memory entries), so a plain locking scheme is plenty.

use std::path::Path;

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::error::{DesktopError, Result};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    User,
    Feedback,
    Project,
    Reference,
}

impl MemoryKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Feedback => "feedback",
            Self::Project => "project",
            Self::Reference => "reference",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "user" => Self::User,
            "feedback" => Self::Feedback,
            "project" => Self::Project,
            "reference" => Self::Reference,
            other => return Err(DesktopError::bad_input(format!("unknown memory kind: {other}"))),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: i64,
    pub kind: MemoryKind,
    pub name: String,
    pub description: String,
    pub body: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUpsert {
    /// Omit for new entries; pass the existing id to update in place.
    pub id: Option<i64>,
    pub kind: MemoryKind,
    pub name: String,
    pub description: String,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: i64,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl MessageRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            other => return Err(DesktopError::bad_input(format!("unknown role: {other}"))),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: i64,
    pub conversation_id: i64,
    pub role: MessageRole,
    pub content: String,
    pub created_at: i64,
}

/// New-image insert payload — passed into `MemoryStore::image_record`.
#[derive(Debug, Clone)]
pub struct ImageRecord<'a> {
    pub path: &'a str,
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub seed: Option<i64>,
    pub steps: Option<i64>,
    pub cfg_scale: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub checkpoint: Option<&'a str>,
    pub conversation_id: Option<i64>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRow {
    pub id: i64,
    pub kind: String,
    pub label: String,
    pub command: String,
    pub status: String,
    /// Free-form JSON dumped from the engine's `JobProgress`.
    pub progress: String,
    pub log_path: String,
    pub started_at: i64,
    pub finished_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRow {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub kind: String,
    pub path: String,
    pub item_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRow {
    pub id: i64,
    pub target_kind: String,
    pub target_id: i64,
    pub rating: i32,
    pub note: Option<String>,
    pub created_at: i64,
}

/// Welford-style running per-class means + the derived steering vector.
/// One row per "user" identity; v1 just uses the literal name `"default"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserVectorState {
    pub name: String,
    pub hidden_size: usize,
    pub n_pos: u32,
    pub n_neg: u32,
    pub mean_pos: Vec<f32>,
    pub mean_neg: Vec<f32>,
    /// `v = mean_pos - mean_neg`. Recomputed on every Welford update.
    pub vec: Vec<f32>,
    /// Multiplier applied at generation time. UI exposes a slider.
    pub alpha: f32,
}

impl UserVectorState {
    pub fn empty(name: &str, hidden_size: usize) -> Self {
        Self {
            name: name.to_string(),
            hidden_size,
            n_pos: 0,
            n_neg: 0,
            mean_pos: vec![0.0; hidden_size],
            mean_neg: vec![0.0; hidden_size],
            vec: vec![0.0; hidden_size],
            alpha: 1.0,
        }
    }

    /// Welford-style update — incrementally tracks the per-class mean
    /// without storing every sample. `sign` is +1 (👍) or -1 (👎).
    pub fn update(&mut self, sign: i32, hidden: &[f32]) {
        if hidden.len() != self.hidden_size {
            // Bail rather than corrupt — caller should have caught this.
            return;
        }
        if sign > 0 {
            self.n_pos += 1;
            let inv = 1.0 / self.n_pos as f32;
            for (m, &h) in self.mean_pos.iter_mut().zip(hidden) {
                *m += (h - *m) * inv;
            }
        } else if sign < 0 {
            self.n_neg += 1;
            let inv = 1.0 / self.n_neg as f32;
            for (m, &h) in self.mean_neg.iter_mut().zip(hidden) {
                *m += (h - *m) * inv;
            }
        }
        // v = μ⁺ − μ⁻
        for ((v, &mp), &mn) in self.vec.iter_mut().zip(&self.mean_pos).zip(&self.mean_neg) {
            *v = mp - mn;
        }
    }

    pub fn norm(&self) -> f32 {
        self.vec.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod welford_tests {
    use super::UserVectorState;

    #[test]
    fn pos_then_neg_yields_difference() {
        let mut s = UserVectorState::empty("u", 3);
        s.update(1, &[1.0, 2.0, 3.0]);
        s.update(1, &[3.0, 2.0, 1.0]);
        // mean_pos = (2.0, 2.0, 2.0)
        s.update(-1, &[0.0, 0.0, 0.0]);
        // mean_neg = (0,0,0); v = (2,2,2)
        assert_eq!(s.n_pos, 2);
        assert_eq!(s.n_neg, 1);
        for &x in &s.vec { assert!((x - 2.0).abs() < 1e-5, "v={:?}", s.vec); }
    }

    #[test]
    fn zero_sign_no_op() {
        let mut s = UserVectorState::empty("u", 2);
        s.update(0, &[1.0, 1.0]);
        assert_eq!(s.n_pos, 0);
        assert_eq!(s.n_neg, 0);
        assert_eq!(s.vec, vec![0.0, 0.0]);
    }

    #[test]
    fn mismatched_hidden_size_bails() {
        let mut s = UserVectorState::empty("u", 4);
        s.update(1, &[1.0, 2.0]);  // wrong size — should be a no-op
        assert_eq!(s.n_pos, 0);
        assert_eq!(s.vec, vec![0.0; 4]);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetUpsert {
    pub id: Option<i64>,
    pub name: String,
    pub description: String,
    pub kind: String,
    pub path: String,
    pub item_count: i64,
}

/// Row returned by `MemoryStore::image_list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredImage {
    pub id: i64,
    pub path: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: Option<i64>,
    pub steps: Option<i64>,
    pub cfg_scale: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub checkpoint: Option<String>,
    pub conversation_id: Option<i64>,
    pub created_at: i64,
}

pub struct MemoryStore {
    conn: Mutex<Connection>,
    /// Path the persistent connection was opened from. Held so we can
    /// reopen it when incognito mode exits. `None` while incognito.
    disk_path: Mutex<Option<std::path::PathBuf>>,
    /// Whether the active connection is the in-memory incognito DB.
    incognito: std::sync::atomic::AtomicBool,
}

impl MemoryStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Self::open_disk_connection(path)?;
        let store = Self {
            conn: Mutex::new(conn),
            disk_path: Mutex::new(Some(path.to_path_buf())),
            incognito: std::sync::atomic::AtomicBool::new(false),
        };
        store.migrate()?;
        Ok(store)
    }

    fn open_disk_connection(path: &Path) -> Result<Connection> {
        let conn = Connection::open(path)
            .map_err(|e| DesktopError::storage(format!("open {}: {e}", path.display())))?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             PRAGMA synchronous = NORMAL;",
        )
        .map_err(|e| DesktopError::storage(format!("pragma: {e}")))?;
        Ok(conn)
    }

    fn open_memory_connection() -> Result<Connection> {
        let conn = Connection::open_in_memory()
            .map_err(|e| DesktopError::storage(format!("open in-memory: {e}")))?;
        // `secure_delete` zeroes deleted page contents inside the DB. For
        // `:memory:` databases this is a defense-in-depth move: when the
        // connection drops the entire arena is freed to the system
        // allocator, but pages we hand-deleted via DELETE / DROP have
        // already been zeroed by SQLite under this pragma.
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA secure_delete = ON;",
        )
        .map_err(|e| DesktopError::storage(format!("incognito pragma: {e}")))?;
        Ok(conn)
    }

    pub fn is_incognito(&self) -> bool {
        self.incognito.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Swap the underlying connection to a fresh in-memory SQLite. The
    /// persistent DB is closed (its connection is dropped) and won't see
    /// any writes until [`exit_incognito`] is called.
    pub fn enter_incognito(&self) -> Result<()> {
        if self.is_incognito() {
            return Ok(());
        }
        let mem_conn = Self::open_memory_connection()?;
        // Remember where to reopen, then swap. The previous connection
        // drops when this guard goes out of scope — closes the disk DB.
        let mut conn = self.conn.lock();
        *conn = mem_conn;
        // Run the schema migration on the fresh in-memory DB.
        drop(conn);
        self.migrate()?;
        self.incognito.store(true, std::sync::atomic::Ordering::SeqCst);
        tracing::info!("MemoryStore entered incognito (in-memory + secure_delete)");
        Ok(())
    }

    /// Swap back to the persistent disk connection. All data accumulated
    /// during the incognito session is lost when the in-memory connection
    /// drops here.
    pub fn exit_incognito(&self) -> Result<()> {
        if !self.is_incognito() {
            return Ok(());
        }
        let disk_path_opt = self.disk_path.lock().clone();
        let disk_path = disk_path_opt
            .ok_or_else(|| DesktopError::storage("no disk path to return to"))?;
        let disk_conn = Self::open_disk_connection(&disk_path)?;
        let mut conn = self.conn.lock();
        *conn = disk_conn;
        drop(conn);
        // The migration was already run when the disk DB was first
        // opened, but re-running is idempotent (CREATE TABLE IF NOT EXISTS).
        self.migrate()?;
        self.incognito.store(false, std::sync::atomic::Ordering::SeqCst);
        tracing::info!(
            path = %disk_path.display(),
            "MemoryStore exited incognito"
        );
        Ok(())
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 kind        TEXT    NOT NULL,
                 name        TEXT    NOT NULL UNIQUE,
                 description TEXT    NOT NULL DEFAULT '',
                 body        TEXT    NOT NULL DEFAULT '',
                 created_at  INTEGER NOT NULL,
                 updated_at  INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

             CREATE TABLE IF NOT EXISTS conversations (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 title       TEXT    NOT NULL,
                 created_at  INTEGER NOT NULL,
                 updated_at  INTEGER NOT NULL
             );

             CREATE TABLE IF NOT EXISTS messages (
                 id              INTEGER PRIMARY KEY AUTOINCREMENT,
                 conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                 role            TEXT    NOT NULL,
                 content         TEXT    NOT NULL,
                 created_at      INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);

             CREATE TABLE IF NOT EXISTS settings (
                 key   TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );

             CREATE TABLE IF NOT EXISTS images (
                 id              INTEGER PRIMARY KEY AUTOINCREMENT,
                 path            TEXT    NOT NULL,
                 prompt          TEXT    NOT NULL,
                 negative_prompt TEXT,
                 seed            INTEGER,
                 steps           INTEGER,
                 cfg_scale       REAL,
                 width           INTEGER,
                 height          INTEGER,
                 checkpoint      TEXT,
                 conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
                 created_at      INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at DESC);

             CREATE TABLE IF NOT EXISTS jobs (
                 id           INTEGER PRIMARY KEY AUTOINCREMENT,
                 kind         TEXT    NOT NULL,
                 label        TEXT    NOT NULL,
                 command      TEXT    NOT NULL,
                 status       TEXT    NOT NULL,
                 progress     TEXT    NOT NULL DEFAULT '{}',
                 log_path     TEXT    NOT NULL,
                 started_at   INTEGER NOT NULL,
                 finished_at  INTEGER
             );
             CREATE INDEX IF NOT EXISTS idx_jobs_started ON jobs(started_at DESC);

             CREATE TABLE IF NOT EXISTS datasets (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 name        TEXT    NOT NULL UNIQUE,
                 description TEXT    NOT NULL DEFAULT '',
                 kind        TEXT    NOT NULL,
                 path        TEXT    NOT NULL,
                 item_count  INTEGER NOT NULL DEFAULT 0,
                 created_at  INTEGER NOT NULL,
                 updated_at  INTEGER NOT NULL
             );

             CREATE TABLE IF NOT EXISTS feedback (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 target_kind TEXT    NOT NULL,
                 target_id   INTEGER NOT NULL,
                 rating      INTEGER NOT NULL,
                 note        TEXT,
                 created_at  INTEGER NOT NULL,
                 UNIQUE(target_kind, target_id)
             );
             CREATE INDEX IF NOT EXISTS idx_feedback_target
                 ON feedback(target_kind, target_id);

             CREATE TABLE IF NOT EXISTS message_hiddens (
                 message_id INTEGER PRIMARY KEY
                              REFERENCES messages(id) ON DELETE CASCADE,
                 hidden     BLOB    NOT NULL,
                 created_at INTEGER NOT NULL
             );

             CREATE TABLE IF NOT EXISTS user_vectors (
                 name        TEXT    PRIMARY KEY,
                 hidden_size INTEGER NOT NULL,
                 n_pos       INTEGER NOT NULL DEFAULT 0,
                 n_neg       INTEGER NOT NULL DEFAULT 0,
                 mean_pos    BLOB    NOT NULL,
                 mean_neg    BLOB    NOT NULL,
                 vec         BLOB    NOT NULL,
                 alpha       REAL    NOT NULL DEFAULT 1.0,
                 updated_at  INTEGER NOT NULL
             );",
        )
        .map_err(|e| DesktopError::storage(format!("migrate: {e}")))?;
        Ok(())
    }

    // ---- memories ----------------------------------------------------------

    pub fn list_memories(&self, kind_filter: Option<MemoryKind>) -> Result<Vec<MemoryEntry>> {
        let conn = self.conn.lock();
        match kind_filter {
            Some(k) => {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, kind, name, description, body, created_at, updated_at
                           FROM memories WHERE kind = ?1 ORDER BY updated_at DESC",
                    )
                    .map_err(DesktopError::from_rusqlite)?;
                let rows = stmt
                    .query_map(params![k.as_str()], row_to_memory)
                    .map_err(DesktopError::from_rusqlite)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(DesktopError::from_rusqlite)
            }
            None => {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, kind, name, description, body, created_at, updated_at
                           FROM memories ORDER BY updated_at DESC",
                    )
                    .map_err(DesktopError::from_rusqlite)?;
                let rows = stmt
                    .query_map([], row_to_memory)
                    .map_err(DesktopError::from_rusqlite)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(DesktopError::from_rusqlite)
            }
        }
    }

    pub fn upsert_memory(&self, m: MemoryUpsert) -> Result<MemoryEntry> {
        let now = now_secs();
        let conn = self.conn.lock();
        let id = match m.id {
            Some(existing) => {
                conn.execute(
                    "UPDATE memories
                        SET kind = ?1, name = ?2, description = ?3, body = ?4, updated_at = ?5
                      WHERE id = ?6",
                    params![m.kind.as_str(), m.name, m.description, m.body, now, existing],
                )
                .map_err(DesktopError::from_rusqlite)?;
                existing
            }
            None => {
                conn.execute(
                    "INSERT INTO memories (kind, name, description, body, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
                    params![m.kind.as_str(), m.name, m.description, m.body, now],
                )
                .map_err(DesktopError::from_rusqlite)?;
                conn.last_insert_rowid()
            }
        };
        // Re-fetch to return the canonical row (handles defaults / triggers).
        let entry = conn
            .query_row(
                "SELECT id, kind, name, description, body, created_at, updated_at
                   FROM memories WHERE id = ?1",
                params![id],
                row_to_memory,
            )
            .map_err(DesktopError::from_rusqlite)?;
        Ok(entry)
    }

    pub fn delete_memory(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock();
        let n = conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .map_err(DesktopError::from_rusqlite)?;
        if n == 0 {
            return Err(DesktopError::bad_input(format!("no memory with id={id}")));
        }
        Ok(())
    }

    // ---- jobs --------------------------------------------------------------

    pub fn job_insert(&self, kind: &str, label: &str, command: &str, log_path: &str) -> Result<i64> {
        let now = now_secs();
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO jobs (kind, label, command, status, progress, log_path, started_at)
             VALUES (?1, ?2, ?3, 'running', '{}', ?4, ?5)",
            params![kind, label, command, log_path, now],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(conn.last_insert_rowid())
    }

    pub fn job_update_progress(&self, id: i64, progress_json: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE jobs SET progress = ?1 WHERE id = ?2",
            params![progress_json, id],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    pub fn job_set_log_path(&self, id: i64, path: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE jobs SET log_path = ?1 WHERE id = ?2",
            params![path, id],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    pub fn job_finish(&self, id: i64, status: &str) -> Result<()> {
        let now = now_secs();
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE jobs SET status = ?1, finished_at = ?2 WHERE id = ?3",
            params![status, now, id],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    pub fn job_list(&self, limit: usize) -> Result<Vec<JobRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, kind, label, command, status, progress, log_path, started_at, finished_at
                   FROM jobs ORDER BY started_at DESC LIMIT ?1",
            )
            .map_err(DesktopError::from_rusqlite)?;
        let rows = stmt
            .query_map(params![limit as i64], row_to_job)
            .map_err(DesktopError::from_rusqlite)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DesktopError::from_rusqlite)
    }

    pub fn job_get(&self, id: i64) -> Result<Option<JobRow>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT id, kind, label, command, status, progress, log_path, started_at, finished_at
               FROM jobs WHERE id = ?1",
            params![id],
            row_to_job,
        )
        .optional()
        .map_err(DesktopError::from_rusqlite)
    }

    /// Called on startup to mark abandoned `running` jobs as aborted.
    /// Subprocesses get reaped when the parent exits (`kill_on_drop`) but the
    /// DB row can't update itself, so this sweep keeps the UI honest.
    pub fn job_reap_orphans(&self) -> Result<()> {
        let conn = self.conn.lock();
        let now = now_secs();
        conn.execute(
            "UPDATE jobs SET status = 'aborted', finished_at = ?1
               WHERE status = 'running'",
            params![now],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    // ---- datasets ----------------------------------------------------------

    pub fn dataset_upsert(&self, d: &DatasetUpsert) -> Result<DatasetRow> {
        let now = now_secs();
        let conn = self.conn.lock();
        let id = match d.id {
            Some(existing) => {
                conn.execute(
                    "UPDATE datasets SET name = ?1, description = ?2, kind = ?3, path = ?4,
                                          item_count = ?5, updated_at = ?6
                       WHERE id = ?7",
                    params![d.name, d.description, d.kind, d.path, d.item_count, now, existing],
                )
                .map_err(DesktopError::from_rusqlite)?;
                existing
            }
            None => {
                conn.execute(
                    "INSERT INTO datasets (name, description, kind, path, item_count, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
                    params![d.name, d.description, d.kind, d.path, d.item_count, now],
                )
                .map_err(DesktopError::from_rusqlite)?;
                conn.last_insert_rowid()
            }
        };
        let row = conn
            .query_row(
                "SELECT id, name, description, kind, path, item_count, created_at, updated_at
                   FROM datasets WHERE id = ?1",
                params![id],
                row_to_dataset,
            )
            .map_err(DesktopError::from_rusqlite)?;
        Ok(row)
    }

    pub fn dataset_list(&self) -> Result<Vec<DatasetRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, kind, path, item_count, created_at, updated_at
                   FROM datasets ORDER BY updated_at DESC",
            )
            .map_err(DesktopError::from_rusqlite)?;
        let rows = stmt
            .query_map([], row_to_dataset)
            .map_err(DesktopError::from_rusqlite)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DesktopError::from_rusqlite)
    }

    pub fn dataset_delete(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM datasets WHERE id = ?1", params![id])
            .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    // ---- feedback ---------------------------------------------------------

    /// Upsert a rating. `rating` is +1 / 0 / -1.
    pub fn feedback_set(
        &self,
        target_kind: &str,
        target_id: i64,
        rating: i32,
        note: Option<&str>,
    ) -> Result<FeedbackRow> {
        let now = now_secs();
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO feedback (target_kind, target_id, rating, note, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(target_kind, target_id) DO UPDATE SET
                 rating = excluded.rating,
                 note   = excluded.note,
                 created_at = excluded.created_at",
            params![target_kind, target_id, rating, note, now],
        )
        .map_err(DesktopError::from_rusqlite)?;
        conn.query_row(
            "SELECT id, target_kind, target_id, rating, note, created_at
               FROM feedback WHERE target_kind = ?1 AND target_id = ?2",
            params![target_kind, target_id],
            row_to_feedback,
        )
        .map_err(DesktopError::from_rusqlite)
    }

    pub fn feedback_get(&self, target_kind: &str, target_id: i64) -> Result<Option<FeedbackRow>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT id, target_kind, target_id, rating, note, created_at
               FROM feedback WHERE target_kind = ?1 AND target_id = ?2",
            params![target_kind, target_id],
            row_to_feedback,
        )
        .optional()
        .map_err(DesktopError::from_rusqlite)
    }

    pub fn feedback_list(
        &self,
        target_kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<FeedbackRow>> {
        let conn = self.conn.lock();
        match target_kind {
            Some(k) => {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, target_kind, target_id, rating, note, created_at
                           FROM feedback WHERE target_kind = ?1
                          ORDER BY created_at DESC LIMIT ?2",
                    )
                    .map_err(DesktopError::from_rusqlite)?;
                let rows = stmt
                    .query_map(params![k, limit as i64], row_to_feedback)
                    .map_err(DesktopError::from_rusqlite)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(DesktopError::from_rusqlite)
            }
            None => {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, target_kind, target_id, rating, note, created_at
                           FROM feedback ORDER BY created_at DESC LIMIT ?1",
                    )
                    .map_err(DesktopError::from_rusqlite)?;
                let rows = stmt
                    .query_map(params![limit as i64], row_to_feedback)
                    .map_err(DesktopError::from_rusqlite)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(DesktopError::from_rusqlite)
            }
        }
    }

    // ---- per-message hidden states ----------------------------------------

    pub fn message_hidden_save(&self, message_id: i64, hidden: &[f32]) -> Result<()> {
        let bytes = f32s_to_bytes(hidden);
        let now = now_secs();
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO message_hiddens (message_id, hidden, created_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(message_id) DO UPDATE SET hidden = excluded.hidden",
            params![message_id, bytes, now],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    pub fn message_hidden_load(&self, message_id: i64) -> Result<Option<Vec<f32>>> {
        let conn = self.conn.lock();
        let row: Option<Vec<u8>> = conn
            .query_row(
                "SELECT hidden FROM message_hiddens WHERE message_id = ?1",
                params![message_id],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(DesktopError::from_rusqlite)?;
        Ok(row.map(|b| bytes_to_f32s(&b)))
    }

    // ---- user steering vector ---------------------------------------------

    pub fn user_vector_load(&self, name: &str, hidden_size: usize) -> Result<UserVectorState> {
        let conn = self.conn.lock();
        let row: Option<(i64, i64, Vec<u8>, Vec<u8>, Vec<u8>, f64)> = conn
            .query_row(
                "SELECT n_pos, n_neg, mean_pos, mean_neg, vec, alpha
                   FROM user_vectors WHERE name = ?1",
                params![name],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
            )
            .optional()
            .map_err(DesktopError::from_rusqlite)?;
        Ok(match row {
            Some((np, nn, mp, mn, vec, alpha)) => UserVectorState {
                name: name.to_string(),
                hidden_size,
                n_pos: np as u32,
                n_neg: nn as u32,
                mean_pos: pad_or_truncate(bytes_to_f32s(&mp), hidden_size),
                mean_neg: pad_or_truncate(bytes_to_f32s(&mn), hidden_size),
                vec: pad_or_truncate(bytes_to_f32s(&vec), hidden_size),
                alpha: alpha as f32,
            },
            None => UserVectorState::empty(name, hidden_size),
        })
    }

    pub fn user_vector_save(&self, s: &UserVectorState) -> Result<()> {
        let now = now_secs();
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO user_vectors
                (name, hidden_size, n_pos, n_neg, mean_pos, mean_neg, vec, alpha, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(name) DO UPDATE SET
                hidden_size = excluded.hidden_size,
                n_pos       = excluded.n_pos,
                n_neg       = excluded.n_neg,
                mean_pos    = excluded.mean_pos,
                mean_neg    = excluded.mean_neg,
                vec         = excluded.vec,
                alpha       = excluded.alpha,
                updated_at  = excluded.updated_at",
            params![
                s.name,
                s.hidden_size as i64,
                s.n_pos as i64,
                s.n_neg as i64,
                f32s_to_bytes(&s.mean_pos),
                f32s_to_bytes(&s.mean_neg),
                f32s_to_bytes(&s.vec),
                s.alpha as f64,
                now,
            ],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    pub fn user_vector_delete(&self, name: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM user_vectors WHERE name = ?1", params![name])
            .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    // ---- conversations ----------------------------------------------------

    pub fn create_conversation(&self, title: &str) -> Result<Conversation> {
        let now = now_secs();
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?1, ?2, ?2)",
            params![title, now],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(Conversation {
            id: conn.last_insert_rowid(),
            title: title.into(),
            created_at: now,
            updated_at: now,
        })
    }

    pub fn list_conversations(&self) -> Result<Vec<Conversation>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, title, created_at, updated_at
                   FROM conversations ORDER BY updated_at DESC",
            )
            .map_err(DesktopError::from_rusqlite)?;
        let rows = stmt
            .query_map([], |r| {
                Ok(Conversation {
                    id: r.get(0)?,
                    title: r.get(1)?,
                    created_at: r.get(2)?,
                    updated_at: r.get(3)?,
                })
            })
            .map_err(DesktopError::from_rusqlite)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DesktopError::from_rusqlite)
    }

    pub fn delete_conversation(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM conversations WHERE id = ?1", params![id])
            .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    pub fn append_message(
        &self,
        conversation_id: i64,
        role: MessageRole,
        content: &str,
    ) -> Result<Message> {
        let now = now_secs();
        let conn = self.conn.lock();
        // Guard against orphan messages — if the conversation was deleted, fail
        // loudly rather than silently inserting a row pointing at nothing.
        let exists = conn
            .query_row(
                "SELECT 1 FROM conversations WHERE id = ?1",
                params![conversation_id],
                |_| Ok(()),
            )
            .optional()
            .map_err(DesktopError::from_rusqlite)?
            .is_some();
        if !exists {
            return Err(DesktopError::bad_input(format!(
                "no conversation with id={conversation_id}"
            )));
        }
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![conversation_id, role.as_str(), content, now],
        )
        .map_err(DesktopError::from_rusqlite)?;
        let id = conn.last_insert_rowid();
        conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now, conversation_id],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(Message {
            id,
            conversation_id,
            role,
            content: content.into(),
            created_at: now,
        })
    }

    // ---- settings (key-value store) ---------------------------------------

    pub fn settings_get(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT value FROM settings WHERE key = ?1",
            params![key],
            |r| r.get::<_, String>(0),
        )
        .optional()
        .map_err(DesktopError::from_rusqlite)
    }

    pub fn settings_set(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(())
    }

    pub fn settings_get_all(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT key, value FROM settings ORDER BY key")
            .map_err(DesktopError::from_rusqlite)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(DesktopError::from_rusqlite)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DesktopError::from_rusqlite)
    }

    // ---- images (gallery + provenance) ------------------------------------

    pub fn image_record(&self, r: &ImageRecord) -> Result<i64> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO images (path, prompt, negative_prompt, seed, steps, cfg_scale,
                                  width, height, checkpoint, conversation_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                r.path, r.prompt, r.negative_prompt, r.seed, r.steps, r.cfg_scale,
                r.width, r.height, r.checkpoint, r.conversation_id, r.created_at,
            ],
        )
        .map_err(DesktopError::from_rusqlite)?;
        Ok(conn.last_insert_rowid())
    }

    pub fn image_list(&self, limit: usize) -> Result<Vec<StoredImage>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, path, prompt, negative_prompt, seed, steps, cfg_scale,
                        width, height, checkpoint, conversation_id, created_at
                   FROM images ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(DesktopError::from_rusqlite)?;
        let rows = stmt
            .query_map(params![limit as i64], |r| {
                Ok(StoredImage {
                    id: r.get(0)?,
                    path: r.get(1)?,
                    prompt: r.get(2)?,
                    negative_prompt: r.get(3)?,
                    seed: r.get(4)?,
                    steps: r.get(5)?,
                    cfg_scale: r.get(6)?,
                    width: r.get(7)?,
                    height: r.get(8)?,
                    checkpoint: r.get(9)?,
                    conversation_id: r.get(10)?,
                    created_at: r.get(11)?,
                })
            })
            .map_err(DesktopError::from_rusqlite)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DesktopError::from_rusqlite)
    }

    /// Fetch a stored image's prompt by id. Used by the CLIP-space feedback
    /// loop to re-encode the prompt when the user rates the image.
    pub fn image_prompt(&self, id: i64) -> Result<Option<String>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT prompt FROM images WHERE id = ?1",
            params![id],
            |r| r.get::<_, String>(0),
        )
        .optional()
        .map_err(DesktopError::from_rusqlite)
    }

    pub fn image_delete(&self, id: i64) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let path: Option<String> = conn
            .query_row("SELECT path FROM images WHERE id = ?1", params![id], |r| r.get(0))
            .optional()
            .map_err(DesktopError::from_rusqlite)?;
        conn.execute("DELETE FROM images WHERE id = ?1", params![id])
            .map_err(DesktopError::from_rusqlite)?;
        Ok(path)
    }

    pub fn conversation_messages(&self, conversation_id: i64) -> Result<Vec<Message>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, conversation_id, role, content, created_at
                   FROM messages WHERE conversation_id = ?1 ORDER BY id ASC",
            )
            .map_err(DesktopError::from_rusqlite)?;
        let rows = stmt
            .query_map(params![conversation_id], |r| {
                let role: String = r.get(2)?;
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    role,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            })
            .map_err(DesktopError::from_rusqlite)?;
        let mut out = Vec::new();
        for row in rows {
            let (id, conv, role, content, ts) = row.map_err(DesktopError::from_rusqlite)?;
            out.push(Message {
                id,
                conversation_id: conv,
                role: MessageRole::parse(&role)?,
                content,
                created_at: ts,
            });
        }
        Ok(out)
    }
}

fn row_to_feedback(r: &rusqlite::Row) -> rusqlite::Result<FeedbackRow> {
    Ok(FeedbackRow {
        id: r.get(0)?,
        target_kind: r.get(1)?,
        target_id: r.get(2)?,
        rating: r.get::<_, i64>(3)? as i32,
        note: r.get(4)?,
        created_at: r.get(5)?,
    })
}

/// Convert a `[f32]` into little-endian bytes for SQLite BLOB storage.
fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
    out
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    let n = b.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        out.push(f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]));
    }
    out
}

/// Older saved vectors may not match a freshly-loaded model's hidden_size.
/// Pad with zeros (a no-op steering direction) or truncate; never silently
/// reinterpret bytes across sizes.
fn pad_or_truncate(mut v: Vec<f32>, target: usize) -> Vec<f32> {
    if v.len() == target { return v; }
    v.resize(target, 0.0);
    v
}

fn row_to_job(r: &rusqlite::Row) -> rusqlite::Result<JobRow> {
    Ok(JobRow {
        id: r.get(0)?,
        kind: r.get(1)?,
        label: r.get(2)?,
        command: r.get(3)?,
        status: r.get(4)?,
        progress: r.get(5)?,
        log_path: r.get(6)?,
        started_at: r.get(7)?,
        finished_at: r.get(8)?,
    })
}

fn row_to_dataset(r: &rusqlite::Row) -> rusqlite::Result<DatasetRow> {
    Ok(DatasetRow {
        id: r.get(0)?,
        name: r.get(1)?,
        description: r.get(2)?,
        kind: r.get(3)?,
        path: r.get(4)?,
        item_count: r.get(5)?,
        created_at: r.get(6)?,
        updated_at: r.get(7)?,
    })
}

fn row_to_memory(r: &rusqlite::Row) -> rusqlite::Result<MemoryEntry> {
    let kind: String = r.get(1)?;
    let kind = MemoryKind::parse(&kind).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(
            1,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::other(e.to_string())),
        )
    })?;
    Ok(MemoryEntry {
        id: r.get(0)?,
        kind,
        name: r.get(2)?,
        description: r.get(3)?,
        body: r.get(4)?,
        created_at: r.get(5)?,
        updated_at: r.get(6)?,
    })
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
