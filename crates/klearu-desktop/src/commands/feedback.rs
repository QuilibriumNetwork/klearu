//! Feedback + user-steering-vector commands.
//!
//! Thumbs-up/down on chat messages do two things atomically:
//!   1. Persist a row in the `feedback` table.
//!   2. If the target is a chat_message AND we have its captured response
//!      hidden, fold the hidden into the per-user Welford state (μ⁺ / μ⁻)
//!      and recompute the steering vector v = μ⁺ − μ⁻.
//!
//! The next `chat_send` reads this vector and routes generation through
//! `generate_streaming_steered`, so the change is visible *the next turn* —
//! that's the "online" in online learning.

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::engine::clip::ClipFormat;
use std::sync::Arc;
use crate::error::{DesktopError, Result};
use crate::memory::{FeedbackRow, UserVectorState};
use crate::state::Workspace;

const DEFAULT_USER: &str = "default";

#[derive(Debug, Deserialize)]
pub struct FeedbackSetArgs {
    /// "chat_message" or "image".
    pub target_kind: String,
    pub target_id: i64,
    /// +1 = 👍, -1 = 👎, 0 = clear.
    pub rating: i32,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FeedbackSetResult {
    pub feedback: FeedbackRow,
    /// When a Welford update fired, contains the user-vector snapshot.
    pub user_vector: Option<UserVectorStatus>,
}

#[tauri::command]
pub async fn feedback_set(
    workspace: State<'_, Arc<Workspace>>,
    args: FeedbackSetArgs,
) -> Result<FeedbackSetResult> {
    let row = workspace
        .memory
        .feedback_set(&args.target_kind, args.target_id, args.rating, args.note.as_deref())?;

    let mut applied_state: Option<UserVectorState> = None;

    // Chat-message feedback → LLM hidden-space Welford update.
    if args.target_kind == "chat_message" && args.rating != 0 {
        if let Some(hidden) = workspace.memory.message_hidden_load(args.target_id)? {
            let hidden_size = hidden.len();
            let mut state = workspace.memory.user_vector_load(DEFAULT_USER, hidden_size)?;
            state.update(args.rating.signum(), &hidden);
            workspace.memory.user_vector_save(&state)?;
            applied_state = Some(state);
        }
    }

    // Image feedback → CLIP-space Welford update. Encodes the image's prompt
    // through CLIP (lazy-loading the engine from current image settings) and
    // folds the resulting per-position-mean into the format-specific user
    // vector (image_clip_sd15 / image_clip_sdxl). Logged-but-silent on
    // failure: feedback rating still persists.
    if args.target_kind == "image" && args.rating != 0 {
        if let Some(prompt) = workspace.memory.image_prompt(args.target_id)? {
            match apply_clip_feedback(&workspace, &prompt, args.rating.signum()).await {
                Ok(state) => applied_state = Some(state),
                Err(e) => {
                    tracing::warn!(?e, "CLIP-space feedback skipped");
                }
            }
        }
    }

    Ok(FeedbackSetResult {
        feedback: row,
        user_vector: applied_state.map(UserVectorStatus::from),
    })
}

/// Lazy-load CLIP from `image.*` settings if needed, encode the prompt,
/// Welford-update the format-keyed user vector, save.
async fn apply_clip_feedback(
    workspace: &State<'_, Arc<Workspace>>,
    prompt: &str,
    sign: i32,
) -> Result<UserVectorState> {
    // Need the diffusion config to know which checkpoint + tokenizers to use.
    let diff_cfg = workspace.diffusion.config();
    let checkpoint = diff_cfg
        .checkpoint
        .clone()
        .ok_or_else(|| DesktopError::bad_input("image.checkpoint not configured"))?;
    let tok_l = diff_cfg
        .tokenizer_l
        .clone()
        .ok_or_else(|| DesktopError::bad_input("image.tokenizer_l not configured"))?;
    let tok_g = diff_cfg.tokenizer_g.clone();

    let clip = workspace.clip.clone();
    // Loading + forward are both compute-heavy: keep them off the event loop.
    let prompt_owned = prompt.to_string();
    let (mean, format) = tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, ClipFormat)> {
        let mut engine = clip.lock();
        let needs_load = match (engine.status().checkpoint.as_deref(), checkpoint.to_str()) {
            (Some(cur), Some(want)) => cur != want,
            _ => true,
        };
        if needs_load {
            engine.load(&checkpoint, &tok_l, tok_g.as_deref())?;
        }
        let mean = engine.encode_mean(&prompt_owned)?;
        let format = engine.format().ok_or_else(|| {
            DesktopError::engine("CLIP loaded but format unknown")
        })?;
        Ok((mean, format))
    })
    .await
    .map_err(|e| DesktopError::engine(format!("join: {e}")))??;

    let key = format.user_vector_key();
    let mut state = workspace.memory.user_vector_load(key, mean.len())?;
    state.update(sign, &mean);
    workspace.memory.user_vector_save(&state)?;
    Ok(state)
}

#[tauri::command]
pub fn feedback_get(
    workspace: State<'_, Arc<Workspace>>,
    target_kind: String,
    target_id: i64,
) -> Result<Option<FeedbackRow>> {
    workspace.memory.feedback_get(&target_kind, target_id)
}

#[derive(Debug, Deserialize)]
pub struct FeedbackListArgs {
    #[serde(default)]
    pub target_kind: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}
fn default_limit() -> usize { 200 }

#[tauri::command]
pub fn feedback_list(
    workspace: State<'_, Arc<Workspace>>,
    args: FeedbackListArgs,
) -> Result<Vec<FeedbackRow>> {
    workspace.memory.feedback_list(args.target_kind.as_deref(), args.limit)
}

// ---- user vector status ----------------------------------------------------

#[derive(Debug, Serialize)]
pub struct UserVectorStatus {
    pub name: String,
    pub hidden_size: usize,
    pub n_pos: u32,
    pub n_neg: u32,
    pub norm: f32,
    pub alpha: f32,
    /// Tiny preview so the UI can show the vector isn't all zero without
    /// shipping the whole hidden_size-element vector.
    pub head: Vec<f32>,
}

impl From<UserVectorState> for UserVectorStatus {
    fn from(s: UserVectorState) -> Self {
        let norm = s.norm();
        let head: Vec<f32> = s.vec.iter().take(8).copied().collect();
        Self {
            name: s.name,
            hidden_size: s.hidden_size,
            n_pos: s.n_pos,
            n_neg: s.n_neg,
            norm,
            alpha: s.alpha,
            head,
        }
    }
}

#[tauri::command]
pub fn user_vector_status(workspace: State<'_, Arc<Workspace>>) -> Result<UserVectorStatus> {
    // We need a hidden_size to read; without a loaded LLM we still answer
    // with whatever's persisted. Engine-loaded size takes precedence — if
    // a model is loaded with a different size, returning padded/truncated
    // is safer than 0.
    let hidden_size = workspace.llm.lock().hidden_size().unwrap_or(0);
    let state = if hidden_size > 0 {
        workspace.memory.user_vector_load(DEFAULT_USER, hidden_size)?
    } else {
        // No LLM loaded: probe whatever shape was last saved by reading
        // a zero-size load and bailing back to empty.
        workspace.memory.user_vector_load(DEFAULT_USER, 1)?
    };
    Ok(UserVectorStatus::from(state))
}

#[derive(Debug, Deserialize)]
pub struct UserVectorAlphaArgs { pub alpha: f32 }

#[tauri::command]
pub fn user_vector_set_alpha(
    workspace: State<'_, Arc<Workspace>>,
    args: UserVectorAlphaArgs,
) -> Result<UserVectorStatus> {
    let hidden_size = workspace.llm.lock().hidden_size().unwrap_or(1);
    let mut state = workspace.memory.user_vector_load(DEFAULT_USER, hidden_size)?;
    state.alpha = args.alpha.clamp(-3.0, 3.0);
    workspace.memory.user_vector_save(&state)?;
    Ok(UserVectorStatus::from(state))
}

#[tauri::command]
pub fn user_vector_reset(workspace: State<'_, Arc<Workspace>>) -> Result<()> {
    workspace.memory.user_vector_delete(DEFAULT_USER)
}

// ---- multi-vector inspection ----------------------------------------------

/// One row per persisted user vector. Returned in alphabetical order by name.
/// `name == "default"` is the LLM chat-space vector; `image_clip_*` rows are
/// the per-format CLIP-space vectors fed by image feedback.
#[tauri::command]
pub fn user_vectors_list(
    workspace: State<'_, Arc<Workspace>>,
) -> Result<Vec<UserVectorStatus>> {
    // We don't have a "select all names" query yet — but the known set is
    // small and stable, so just probe the namespace explicitly.
    let names: &[(&str, usize)] = &[
        ("default", workspace.llm.lock().hidden_size().unwrap_or(1)),
        ("image_clip_sd15", 768),
        ("image_clip_sdxl", 2048),
    ];
    let mut out = Vec::with_capacity(names.len());
    for (name, default_size) in names {
        let state = workspace.memory.user_vector_load(name, *default_size)?;
        // Only surface rows that actually carry signal (or have an
        // explicit alpha override — user may have nudged the slider
        // before any feedback came in).
        if state.n_pos + state.n_neg > 0 || (state.alpha != 1.0 && state.alpha != 0.0) {
            out.push(UserVectorStatus::from(state));
        }
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
pub struct UserVectorNameArgs { pub name: String }

#[tauri::command]
pub fn user_vector_reset_named(
    workspace: State<'_, Arc<Workspace>>,
    args: UserVectorNameArgs,
) -> Result<()> {
    workspace.memory.user_vector_delete(&args.name)
}

#[derive(Debug, Deserialize)]
pub struct UserVectorAlphaNamedArgs { pub name: String, pub alpha: f32 }

#[tauri::command]
pub fn user_vector_set_alpha_named(
    workspace: State<'_, Arc<Workspace>>,
    args: UserVectorAlphaNamedArgs,
) -> Result<UserVectorStatus> {
    // hidden_size is whatever's already persisted — fetch the existing row
    // first so we don't accidentally resize a non-empty vector.
    let existing = workspace.memory.user_vector_load(&args.name, 1)?;
    let mut state = workspace.memory.user_vector_load(&args.name, existing.hidden_size.max(1))?;
    state.alpha = args.alpha.clamp(-3.0, 3.0);
    workspace.memory.user_vector_save(&state)?;
    Ok(UserVectorStatus::from(state))
}
