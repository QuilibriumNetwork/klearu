//! Generic long-running-subprocess manager.
//!
//! Used for training (`img_train`, future LLM finetune), distillation
//! (`scripts/distill_from_sdxl.py`), and any other "kick this off and tell
//! me when it's done" workflow. Three responsibilities:
//!
//!   1. Spawn a `Command`, redirect stdout+stderr to a log file, persist a
//!      `jobs` row.
//!   2. While running, parse interesting stderr lines into a `JobProgress`
//!      struct and dump it back to SQLite so polling clients (the UI) see
//!      live progress.
//!   3. On exit, mark the row `completed` / `failed` / `aborted`.
//!
//! No Tauri Channel — polling (`job_get`) is simpler and survives the user
//! switching tabs. The log file is the source of truth for the live tail.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::error::{DesktopError, Result};
use crate::memory::MemoryStore;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobProgress {
    /// Current step / epoch / sample.
    pub step: Option<u64>,
    /// Announced total when known.
    pub total: Option<u64>,
    /// Most recent loss / training metric.
    pub loss: Option<f32>,
    /// Wall-clock seconds since job started.
    pub elapsed_s: Option<f32>,
    /// Free-form: the parser stashes anything else here. The UI renders
    /// keys it knows about, ignores the rest.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One pluggable parser per job kind. Each call gets a stderr/stdout line
/// and updates the shared progress struct in place.
pub type ParseFn = fn(&str, &mut JobProgress);

struct JobHandle {
    abort: mpsc::Sender<()>,
}

#[derive(Default, Clone)]
pub struct JobManager {
    inner: Arc<JobManagerInner>,
}

#[derive(Default)]
struct JobManagerInner {
    jobs: Mutex<HashMap<i64, JobHandle>>,
}

impl JobManager {
    pub fn new() -> Self { Self::default() }

    /// Spawn `cmd` as a tracked job. Returns the row id.
    ///
    /// `parser` is called per stderr line; mutate the shared progress and
    /// it'll be flushed to SQLite every flush_window lines.
    pub async fn spawn(
        &self,
        memory: Arc<MemoryStore>,
        data_dir: PathBuf,
        kind: &str,
        label: &str,
        mut cmd: Command,
        parser: ParseFn,
    ) -> Result<i64> {
        let jobs_dir = data_dir.join("jobs");
        std::fs::create_dir_all(&jobs_dir)
            .map_err(|e| DesktopError::Io(format!("create {}: {e}", jobs_dir.display())))?;

        // Render the command line for the audit trail. tokio::Command doesn't
        // expose the program/argv pair directly, so we let the user assemble
        // the label themselves and snapshot whatever `Debug` shows.
        let command_str = format!("{:?}", cmd.as_std());

        // Allocate the row id first so the log filename can include it.
        // Use a placeholder log path; we'll patch it after we know the id.
        let placeholder = jobs_dir.join("pending.log");
        let id = memory.job_insert(kind, label, &command_str, placeholder.to_str().unwrap_or(""))?;
        let log_path = jobs_dir.join(format!("{id}.log"));
        // Update the row with the final log path (uses settings_set-style
        // upsert pattern; cheap).
        memory.job_set_log_path(id, &log_path.to_string_lossy())?;

        let log_file = std::fs::File::create(&log_path)
            .map_err(|e| DesktopError::Io(format!("create log {}: {e}", log_path.display())))?;
        let log_for_stderr = log_file
            .try_clone()
            .map_err(|e| DesktopError::Io(format!("dup log fd: {e}")))?;
        cmd.stdout(Stdio::from(log_file));
        cmd.stderr(Stdio::piped()); // we tee stderr → log + parser
        cmd.kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| DesktopError::engine(format!("spawn job: {e}")))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| DesktopError::engine("no stderr pipe"))?;

        let (abort_tx, abort_rx) = mpsc::channel::<()>(1);
        self.inner.jobs.lock().insert(id, JobHandle { abort: abort_tx });

        // Drive parsing + persistence on a tokio task. The job outlives this
        // function: the user can navigate away, switch tabs, etc.
        let memory_for_task = memory.clone();
        let inner = self.inner.clone();
        let log_path_for_task = log_path.clone();
        tokio::spawn(async move {
            let res = run_job_loop(
                child, stderr, log_for_stderr, abort_rx, memory_for_task.clone(),
                id, parser, log_path_for_task,
            ).await;
            inner.jobs.lock().remove(&id);
            // Final status — preserves whatever run_job_loop set last (the
            // status defaults to whatever Result yielded).
            if let Err(e) = res {
                tracing::warn!(job_id = id, ?e, "job loop returned error");
                let _ = memory_for_task.job_finish(id, "failed");
            }
        });

        Ok(id)
    }

    pub fn abort(&self, id: i64) -> bool {
        if let Some(h) = self.inner.jobs.lock().get(&id) {
            let _ = h.abort.try_send(());
            tracing::info!(job_id = id, "abort requested");
            true
        } else {
            false
        }
    }
}


async fn run_job_loop(
    mut child: Child,
    stderr: tokio::process::ChildStderr,
    mut log_for_stderr: std::fs::File,
    mut abort_rx: mpsc::Receiver<()>,
    memory: Arc<MemoryStore>,
    job_id: i64,
    parser: ParseFn,
    log_path: PathBuf,
) -> Result<()> {
    use std::io::Write;

    let started = std::time::Instant::now();
    let progress = Arc::new(Mutex::new(JobProgress::default()));

    let reader = BufReader::new(stderr);
    let mut lines = reader.lines();

    // Flush progress to SQLite every N lines, otherwise the UI sees stale
    // numbers; updating per-line would be wasteful for chatty trainers.
    const FLUSH_EVERY: usize = 8;
    let mut lines_since_flush = 0usize;

    loop {
        tokio::select! {
            line = lines.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        // Tee to log file.
                        let _ = writeln!(log_for_stderr, "{line}");
                        // Parse → progress.
                        let mut p = progress.lock();
                        parser(&line, &mut p);
                        p.elapsed_s = Some(started.elapsed().as_secs_f32());
                        let json = serde_json::to_string(&*p).unwrap_or_else(|_| "{}".into());
                        drop(p);
                        lines_since_flush += 1;
                        if lines_since_flush >= FLUSH_EVERY {
                            lines_since_flush = 0;
                            let _ = memory.job_update_progress(job_id, &json);
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::warn!(?e, job_id, "stderr read error");
                        break;
                    }
                }
            }
            _ = abort_rx.recv() => {
                let _ = child.start_kill();
                // Drain remaining stderr quickly so the log file is complete.
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = writeln!(log_for_stderr, "{line}");
                }
                let _ = child.wait().await;
                let _ = memory.job_finish(job_id, "aborted");
                let _ = log_path; // silence unused warning when no debug log
                return Ok(());
            }
        }
    }

    // Final progress flush.
    {
        let p = progress.lock();
        let json = serde_json::to_string(&*p).unwrap_or_else(|_| "{}".into());
        let _ = memory.job_update_progress(job_id, &json);
    }

    let status = child
        .wait()
        .await
        .map_err(|e| DesktopError::engine(format!("wait: {e}")))?;
    let final_status = if status.success() { "completed" } else { "failed" };
    let _ = memory.job_finish(job_id, final_status);
    Ok(())
}

// ---- progress parsers ------------------------------------------------------

/// Parser for `img_train` log lines. Recognizes:
///   `[img_train] step N/M  loss=X.XXXX  (Y.Ys elapsed, Z.Zs/step)`
pub fn parse_img_train(line: &str, p: &mut JobProgress) {
    let Some(rest) = line.strip_prefix("[img_train] step ") else { return; };
    let slash = match rest.find('/') { Some(i) => i, None => return };
    let after_slash = &rest[slash + 1..];
    let space = match after_slash.find(|c: char| !c.is_ascii_digit()) {
        Some(i) => i, None => return
    };
    if let Ok(step) = rest[..slash].parse::<u64>() {
        p.step = Some(step);
    }
    if let Ok(total) = after_slash[..space].parse::<u64>() {
        p.total = Some(total);
    }
    if let Some(loss_idx) = line.find("loss=") {
        let tail = &line[loss_idx + 5..];
        let end = tail.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(tail.len());
        if let Ok(loss) = tail[..end].parse::<f32>() {
            p.loss = Some(loss);
        }
    }
}

/// Parser for `scripts/distill_from_sdxl.py`. Recognizes:
///   `[distill] prompts=N/M ...`
///   `[distill] dropped N (last: REASON)`
pub fn parse_distill(line: &str, p: &mut JobProgress) {
    if let Some(rest) = line.strip_prefix("[distill] prompts=") {
        if let Some(slash) = rest.find('/') {
            if let Ok(n) = rest[..slash].parse::<u64>() {
                p.step = Some(n);
            }
            let after = &rest[slash + 1..];
            let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
            if let Ok(t) = after[..end].parse::<u64>() {
                p.total = Some(t);
            }
        }
    }
    if let Some(rest) = line.strip_prefix("[distill] dropped ") {
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        if let Ok(n) = rest[..end].parse::<u64>() {
            p.extra.insert("dropped".into(), serde_json::Value::from(n));
        }
    }
}

/// Read the last `max_lines` lines of a log file. The job-detail panel
/// uses this for a live tail without sucking the whole file into memory
/// for long-running trainers.
pub fn tail_log(path: &Path, max_lines: usize) -> Result<Vec<String>> {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    const READ_BACK: u64 = 32 * 1024;

    let mut f = std::fs::File::open(path)
        .map_err(|e| DesktopError::Io(format!("open log {}: {e}", path.display())))?;
    let len = f.seek(SeekFrom::End(0)).map_err(io_err)?;
    // Read up to READ_BACK bytes from end; on small files just read the whole thing.
    let start = len.saturating_sub(READ_BACK);
    f.seek(SeekFrom::Start(start)).map_err(io_err)?;
    let reader = BufReader::new(f);
    let lines: Vec<String> = reader.lines().map_while(|l| l.ok()).collect();
    let n = lines.len();
    let skip = n.saturating_sub(max_lines);
    Ok(lines.into_iter().skip(skip).collect())
}

fn io_err(e: std::io::Error) -> DesktopError {
    DesktopError::Io(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_img_train_line() {
        let mut p = JobProgress::default();
        parse_img_train(
            "[img_train] step 250/10000  loss=4.2138  (12.4s elapsed, 0.05s/step)",
            &mut p,
        );
        assert_eq!(p.step, Some(250));
        assert_eq!(p.total, Some(10000));
        assert!((p.loss.unwrap() - 4.2138).abs() < 1e-3);
    }

    #[test]
    fn parses_distill_line() {
        let mut p = JobProgress::default();
        parse_distill("[distill] prompts=3/100 some other text", &mut p);
        assert_eq!(p.step, Some(3));
        assert_eq!(p.total, Some(100));
    }

    #[test]
    fn ignores_irrelevant_lines() {
        let mut p = JobProgress::default();
        parse_img_train("[img_train] saving checkpoint…", &mut p);
        assert_eq!(p.step, None);
    }
}
