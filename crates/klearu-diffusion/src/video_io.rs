//! Video I/O — encode a sequence of CHW f32 frames into an MP4 via ffmpeg.
//!
//! We shell out to ffmpeg rather than linking libavcodec because:
//!   - ffmpeg is universally installed (`brew install ffmpeg`, apt, etc)
//!   - The pipe-based approach is incognito-safe: frames flow from memory
//!     to ffmpeg's stdin; the output can go to a file OR to stdout and
//!     be captured as bytes.
//!   - No C-binding maintenance burden.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::error::{DiffusionError, Result};
use crate::pipeline::GeneratedImage;

/// Encode `frames` into an H.264 MP4 at the given FPS.
///
/// When `out_path` is `Some`, writes to that file. When `None`, captures
/// the MP4 bytes in memory (for incognito / data-URI delivery).
pub fn encode_mp4(
    frames: &[GeneratedImage],
    fps: usize,
    out_path: Option<&Path>,
) -> Result<Vec<u8>> {
    if frames.is_empty() {
        return Err(DiffusionError::Unsupported("no frames to encode".into()));
    }
    let h = frames[0].height;
    let w = frames[0].width;

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y") // overwrite
        .arg("-f").arg("rawvideo")
        .arg("-pixel_format").arg("rgb24")
        .arg("-video_size").arg(format!("{w}x{h}"))
        .arg("-framerate").arg(fps.to_string())
        .arg("-i").arg("pipe:0") // read raw frames from stdin
        .arg("-c:v").arg("libx264")
        .arg("-pix_fmt").arg("yuv420p")
        .arg("-crf").arg("18") // visually lossless
        .arg("-preset").arg("fast")
        .arg("-movflags").arg("+faststart"); // streaming-friendly

    if let Some(p) = out_path {
        cmd.arg("-movflags").arg("+faststart");
        cmd.arg(p);
        cmd.stdout(Stdio::null());
    } else {
        // Pipe to stdout: use fragmented MP4 which doesn't need seeking
        // (regular MP4 requires a seekable output for the moov atom).
        cmd.arg("-movflags").arg("frag_keyframe+empty_moov+default_base_moof");
        cmd.arg("-f").arg("mp4").arg("pipe:1");
        cmd.stdout(Stdio::piped());
    }
    cmd.stdin(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        DiffusionError::Unsupported(format!(
            "ffmpeg not found — install it (`brew install ffmpeg` / `apt install ffmpeg`): {e}"
        ))
    })?;

    // Write raw RGB frames to ffmpeg's stdin.
    let write_result = {
        let stdin = child.stdin.as_mut()
            .ok_or_else(|| DiffusionError::Unsupported("ffmpeg stdin pipe missing".into()))?;
        let mut err = None;
        for frame in frames {
            let rgb_bytes = chw_f32_to_rgb_u8(&frame.rgb_chw, frame.height, frame.width);
            if let Err(e) = stdin.write_all(&rgb_bytes) {
                err = Some(e);
                break;
            }
        }
        err
    }; // stdin drops here → ffmpeg sees EOF and finishes encoding

    let output = child.wait_with_output()
        .map_err(|e| DiffusionError::Unsupported(format!("ffmpeg wait: {e}")))?;

    if !output.status.success() || write_result.is_some() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let pipe_err = write_result.map(|e| format!(" (pipe: {e})")).unwrap_or_default();
        return Err(DiffusionError::Unsupported(format!(
            "ffmpeg exited with {}{}:\n{}",
            output.status.code().unwrap_or(-1),
            pipe_err,
            stderr.lines().rev().take(5).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n"),
        )));
    }

    if out_path.is_some() {
        Ok(Vec::new()) // file written directly by ffmpeg
    } else {
        Ok(output.stdout) // MP4 bytes captured from stdout
    }
}

/// Convert a `[3, H, W]` f32 CHW buffer in `[-1, 1]` to interleaved
/// `[H, W, 3]` u8 RGB. Same conversion as `image_io::save_png` but
/// returns raw bytes for piping to ffmpeg.
fn chw_f32_to_rgb_u8(chw: &[f32], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for hi in 0..h {
        for wi in 0..w {
            for c in 0..3 {
                let v = chw[c * h * w + hi * w + wi];
                let v01 = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
                out[(hi * w + wi) * 3 + c] = (v01 * 255.0).round() as u8;
            }
        }
    }
    out
}
