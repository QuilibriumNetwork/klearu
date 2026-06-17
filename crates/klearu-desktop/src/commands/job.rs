//! Job commands — start (training, distillation), list, abort, tail log.
//!
//! Heavy work runs as subprocesses tracked by [`JobManager`]. UI polls via
//! `job_get` for progress; the log tail is `job_log_tail`.

use std::path::PathBuf;

use serde::Deserialize;
use tauri::State;
use tokio::process::Command;

use crate::engine::jobs::{parse_distill, parse_img_train, tail_log};
use std::sync::Arc;
use crate::error::{DesktopError, Result};
use crate::memory::JobRow;
use crate::state::Workspace;

#[tauri::command]
pub fn job_list(workspace: State<'_, Arc<Workspace>>, limit: Option<usize>) -> Result<Vec<JobRow>> {
    workspace.memory.job_list(limit.unwrap_or(50))
}

#[tauri::command]
pub fn job_get(workspace: State<'_, Arc<Workspace>>, id: i64) -> Result<Option<JobRow>> {
    workspace.memory.job_get(id)
}

#[tauri::command]
pub fn job_abort(workspace: State<'_, Arc<Workspace>>, id: i64) -> bool {
    workspace.jobs.abort(id)
}

#[derive(Debug, Deserialize)]
pub struct JobLogTailArgs {
    pub id: i64,
    #[serde(default = "default_log_lines")]
    pub max_lines: usize,
}
fn default_log_lines() -> usize { 200 }

#[tauri::command]
pub fn job_log_tail(
    workspace: State<'_, Arc<Workspace>>,
    args: JobLogTailArgs,
) -> Result<Vec<String>> {
    let Some(row) = workspace.memory.job_get(args.id)? else {
        return Err(DesktopError::bad_input(format!("no job with id={}", args.id)));
    };
    let p = PathBuf::from(&row.log_path);
    if !p.exists() {
        return Ok(Vec::new());
    }
    tail_log(&p, args.max_lines)
}

// ---- launch: img_train ----------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ImgTrainArgs {
    pub manifest: String,
    pub out: String,
    pub steps: u64,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_log_every")]
    pub log_every: u64,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub resume: Option<String>,
    #[serde(default)]
    pub bpe: Option<String>,
    /// Path to the `img_train` binary. Falls back to `target/{release,debug}/img_train`.
    #[serde(default)]
    pub binary_path: Option<String>,
    /// Optional user-supplied label for the jobs list.
    #[serde(default)]
    pub label: Option<String>,
}
fn default_lr() -> f32 { 3e-4 }
fn default_log_every() -> u64 { 100 }

#[tauri::command]
pub async fn job_train_image(
    workspace: State<'_, Arc<Workspace>>,
    args: ImgTrainArgs,
) -> Result<i64> {
    workspace.require_not_incognito("image training (writes checkpoint + logs to disk)")?;
    let binary = resolve_binary(args.binary_path.as_deref(), "img_train", "KLEARU_IMG_TRAIN")?;
    let mut cmd = Command::new(&binary);
    cmd.arg("--manifest").arg(&args.manifest);
    cmd.arg("--out").arg(&args.out);
    cmd.arg("--steps").arg(args.steps.to_string());
    cmd.arg("--lr").arg(args.lr.to_string());
    cmd.arg("--log-every").arg(args.log_every.to_string());
    if let Some(s) = args.seed { cmd.arg("--seed").arg(s.to_string()); }
    if let Some(r) = args.resume.as_deref() { cmd.arg("--resume").arg(r); }
    if let Some(b) = args.bpe.as_deref() { cmd.arg("--bpe").arg(b); }

    let memory = workspace.memory.clone();
    let data_dir = workspace.data_dir();
    let label = args.label.unwrap_or_else(|| format!("img_train · {} steps", args.steps));
    workspace
        .jobs
        .spawn(memory, data_dir, "train_image", &label, cmd, parse_img_train)
        .await
}

// ---- launch: distill (Python subprocess) ----------------------------------

#[derive(Debug, Deserialize)]
pub struct DistillArgs {
    pub prompts: String,
    pub sdxl: String,
    pub out_dir: String,
    #[serde(default)]
    pub planner_bin: Option<String>,
    #[serde(default)]
    pub planner_args: Option<String>,
    #[serde(default)]
    pub davit_model: Option<String>,
    #[serde(default)]
    pub davit_conf: Option<f32>,
    #[serde(default)]
    pub n_images: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub guidance: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_prompts: Option<u32>,
    #[serde(default)]
    pub label: Option<String>,
    /// Path to python. Defaults to `python3`, falls back to `python`.
    #[serde(default)]
    pub python: Option<String>,
    /// Path to `scripts/distill_from_sdxl.py`. Tries to auto-locate when omitted.
    #[serde(default)]
    pub script: Option<String>,
}

#[tauri::command]
pub async fn job_distill(
    workspace: State<'_, Arc<Workspace>>,
    args: DistillArgs,
) -> Result<i64> {
    workspace.require_not_incognito("distillation")?;
    let python = args.python.clone().unwrap_or_else(|| "python3".into());
    let script = resolve_distill_script(args.script.as_deref())?;
    let mut cmd = Command::new(&python);
    cmd.arg(&script);
    cmd.arg("--prompts").arg(&args.prompts);
    cmd.arg("--sdxl").arg(&args.sdxl);
    cmd.arg("--out-dir").arg(&args.out_dir);
    if let Some(p) = &args.planner_bin { cmd.arg("--planner-bin").arg(p); }
    if let Some(pa) = &args.planner_args { cmd.arg("--planner-args").arg(pa); }
    if let Some(d) = &args.davit_model { cmd.arg("--davit-model").arg(d); }
    if let Some(c) = args.davit_conf { cmd.arg("--davit-conf").arg(c.to_string()); }
    if let Some(n) = args.n_images { cmd.arg("--n-images").arg(n.to_string()); }
    if let Some(s) = args.steps { cmd.arg("--steps").arg(s.to_string()); }
    if let Some(g) = args.guidance { cmd.arg("--guidance").arg(g.to_string()); }
    if let Some(s) = args.seed { cmd.arg("--seed").arg(s.to_string()); }
    if let Some(m) = args.max_prompts { cmd.arg("--max-prompts").arg(m.to_string()); }

    let memory = workspace.memory.clone();
    let data_dir = workspace.data_dir();
    let label = args.label.unwrap_or_else(|| format!("distill · {}", args.out_dir));
    workspace
        .jobs
        .spawn(memory, data_dir, "distill", &label, cmd, parse_distill)
        .await
}

// ---- helpers ---------------------------------------------------------------

fn resolve_binary(
    override_path: Option<&str>,
    bin_name: &str,
    env_var: &str,
) -> Result<PathBuf> {
    if let Some(p) = override_path {
        let p = PathBuf::from(p);
        if p.exists() { return Ok(p); }
        return Err(DesktopError::bad_input(format!("{bin_name} not found at {}", p.display())));
    }
    if let Ok(env) = std::env::var(env_var) {
        let p = PathBuf::from(env);
        if p.exists() { return Ok(p); }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let sibling = parent.join(bin_name);
            if sibling.exists() { return Ok(sibling); }
            for ancestor in parent.ancestors().take(4) {
                for profile in ["release", "debug"] {
                    let candidate = ancestor.join("target").join(profile).join(bin_name);
                    if candidate.exists() { return Ok(candidate); }
                }
            }
        }
    }
    if let Ok(out) = std::process::Command::new("which").arg(bin_name).output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let p = PathBuf::from(s);
            if p.exists() { return Ok(p); }
        }
    }
    Err(DesktopError::bad_input(format!(
        "{bin_name} not found. Set {env_var} or pass `binary_path`."
    )))
}

fn resolve_distill_script(override_path: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        let p = PathBuf::from(p);
        if p.exists() { return Ok(p); }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for ancestor in parent.ancestors().take(6) {
                let candidate = ancestor.join("scripts").join("distill_from_sdxl.py");
                if candidate.exists() { return Ok(candidate); }
            }
        }
    }
    Err(DesktopError::bad_input(
        "scripts/distill_from_sdxl.py not found — pass `script` explicitly.",
    ))
}
