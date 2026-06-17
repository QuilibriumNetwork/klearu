//! Long-lived application state held in Tauri's `State<>` slot.
//!
//! `Workspace` is constructed once per app launch. Engines start unloaded —
//! the LLM checkpoint is gigabytes, the diffusion checkpoint is many more,
//! and resident-by-default would be hostile on smaller machines. Each
//! engine is hidden behind a `parking_lot::Mutex` because the underlying
//! `Pipeline` mutates its KV caches during generation.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::engine::{ClipEngine, DiffusionEngine, JobManager, LlmEngine, VisionEngine};
use crate::error::{DesktopError, Result};
use crate::memory::MemoryStore;

/// Active incognito session metadata. Created when the user toggles
/// incognito on; dropped on toggle off.
///
/// There is no scratch directory and no on-disk artifacts of any kind.
/// Features that can't operate without a filesystem path (image
/// generation via the `sd_generate` subprocess, distillation jobs) are
/// refused while a session is active — they'll be re-enabled once the
/// diffusion library refactor (Refactors 2+3) makes them callable
/// in-process with bytes instead of paths.
pub struct IncognitoSession {
    pub started_at: i64,
}

pub struct Workspace {
    pub data_dir: RwLock<PathBuf>,
    pub memory: Arc<MemoryStore>,
    pub llm: Arc<Mutex<LlmEngine>>,
    pub diffusion: DiffusionEngine,
    pub vision: Arc<Mutex<VisionEngine>>,
    pub clip: Arc<Mutex<ClipEngine>>,
    pub jobs: JobManager,
    pub incognito: RwLock<Option<Arc<IncognitoSession>>>,
    /// Active LAN server handle — dropped on stop.
    pub lan_handle: Mutex<Option<crate::lan_server::LanHandle>>,
}

impl Workspace {
    pub fn new() -> Result<Self> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let data_dir = home.join(".klearu");
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| DesktopError::Io(format!("create {}: {e}", data_dir.display())))?;
        // Pre-create the images subdir so the gallery panel never sees a
        // missing-dir error before the first generation.
        let _ = std::fs::create_dir_all(data_dir.join("images"));
        let memory = Arc::new(MemoryStore::open(&data_dir.join("klearu.sqlite"))?);
        let llm = Arc::new(Mutex::new(LlmEngine::new()));
        let diffusion = DiffusionEngine::new();
        // Hydrate diffusion config from settings if anything was saved.
        diffusion.set_config(diffusion_config_from_settings(&memory));
        let vision = Arc::new(Mutex::new(VisionEngine::new()));
        let clip = Arc::new(Mutex::new(ClipEngine::new()));
        // Any `running` job rows are leftovers from a previous run —
        // their children died with the parent. Mark them aborted so the
        // UI's "running" list isn't lying.
        let _ = memory.job_reap_orphans();
        let jobs = JobManager::new();
        Ok(Self {
            data_dir: RwLock::new(data_dir),
            memory,
            llm,
            diffusion,
            vision,
            clip,
            jobs,
            incognito: RwLock::new(None),
            lan_handle: Mutex::new(None),
        })
    }

    pub fn data_dir(&self) -> PathBuf { self.data_dir.read().clone() }

    pub fn is_incognito(&self) -> bool { self.incognito.read().is_some() }

    /// Directory for ephemeral artifacts (generated images, cond-bias
    /// sidecars, inpaint inputs). Returns the regular data dir. The
    /// caller MUST also check [`Self::is_incognito`] and refuse the
    /// operation outright if it can't avoid a disk touch — there is no
    /// memory-backed scratch path while incognito is active.
    pub fn writes_dir(&self) -> PathBuf {
        self.data_dir()
    }

    /// Convenience: fail the current command if incognito is active. Use
    /// from any command that needs to write to disk (image generation,
    /// distillation, etc).
    pub fn require_not_incognito(&self, feature: &str) -> Result<()> {
        if self.is_incognito() {
            return Err(DesktopError::bad_input(format!(
                "{feature} writes files to disk — disabled while incognito is active"
            )));
        }
        Ok(())
    }
}

fn diffusion_config_from_settings(memory: &MemoryStore) -> crate::engine::diffusion::DiffusionConfig {
    use crate::engine::diffusion::DiffusionConfig;
    let get = |k: &str| memory.settings_get(k).ok().flatten();
    DiffusionConfig {
        sd_generate_path: get("image.sd_generate_path").map(PathBuf::from),
        checkpoint: get("image.checkpoint").map(PathBuf::from),
        tokenizer_l: get("image.tokenizer_l").map(PathBuf::from),
        tokenizer_g: get("image.tokenizer_g").map(PathBuf::from),
        output_dir: get("image.output_dir").map(PathBuf::from),
        motion_adapter: get("image.motion_adapter").map(PathBuf::from),
    }
}
