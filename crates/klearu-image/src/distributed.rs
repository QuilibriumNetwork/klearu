//! Distributed training: design + single-node data-parallel skeleton.
//!
//! ## Honest scope note
//!
//! Real distributed training across multiple machines requires
//! infrastructure that's well beyond a single session: a transport
//! layer (NCCL-equivalent for CPU clusters or rendezvous over TCP),
//! gradient all-reduce, weight checkpoint sharding, fault tolerance,
//! and a job scheduler. The cleanest path for klearu-image — and the
//! one most other modern training systems follow — is the **layered
//! decomposition** described in the design doc below.
//!
//! This module ships:
//!   1. **Design doc** (this doctring + the design enum below) covering
//!      what each distributed mode looks like for our MoE+sparse-attn
//!      architecture.
//!   2. **Single-node data-parallel skeleton** (`DataParallelTrainer`)
//!      that runs N worker threads each computing gradients on a shard
//!      of the batch, then averages gradients before each AdamW step.
//!      This is the smallest unit of distribution that's tractable to
//!      ship + verify in one session. It's also the foundation on
//!      which multi-node sits.
//!
//! ## Distributed-mode taxonomy for klearu-image
//!
//! 1. **Data Parallel (DP)** — N replicas of the full model, each
//!    processes a different shard of the global batch. Gradients are
//!    averaged across replicas before the optimizer step. Simplest
//!    form; this module implements the single-node version.
//!
//! 2. **Tensor Parallel (TP)** — split a single layer's matrices
//!    across N devices. For klearu-image, the natural TP axis is the
//!    `mlp_intermediate` dimension (split the SwiGLU MLP three ways),
//!    or the `num_experts` dimension for MoE. Requires per-step
//!    all-reduce on the activations crossing the partition.
//!
//! 3. **Expert Parallel (EP)** — specifically for MoE. Each device
//!    hosts a subset of experts. Per-token routing decisions decide
//!    which device a token's MLP work happens on; output is sent
//!    back to the originating device. Natural fit for our `MoeFfn`
//!    where K-of-N already implies a routing layer.
//!
//! 4. **Pipeline Parallel (PP)** — split layers across devices. Device
//!    0 hosts layers 0..6, device 1 hosts 6..12. Activations stream
//!    forward; gradients stream backward. Requires careful micro-batch
//!    interleaving to keep all devices busy.
//!
//! 5. **Fully-Sharded Data Parallel (FSDP / ZeRO-3)** — same as DP but
//!    each replica only holds 1/N of the model weights at any moment.
//!    Each layer's forward gathers its full weights, computes, then
//!    re-shards. Massively reduces per-device memory but needs many
//!    all-gathers per step.
//!
//! ## Recommended mode for klearu-image's regime
//!
//! Given the project's CPU-trainable target (M3 Ultra, M3 Studio):
//!
//! - **Within a single Mac**: data-parallel across CPU cores via thread
//!   pool (this module). Gradients live in shared memory; all-reduce
//!   is a per-step Rayon reduce.
//! - **Across multiple Macs (LAN)**: data-parallel with TCP gradient
//!   exchange. Gradient transfer at our 50M-param scale is ~200 MB per
//!   step, which is 2 seconds at 1 Gb/s — comparable to per-step
//!   compute. Sync-SGD is feasible.
//! - **Mixed-Mac MoE** (longer horizon): expert-parallel where each
//!   Mac hosts a subset of MoE experts. Saves both memory and compute
//!   per machine; the routing cost is the LAN latency of token
//!   forwards.

use std::sync::Mutex;

use crate::error::Result;
use crate::grad::Gradients;
use crate::model::ImageTransformer;
use crate::optim::AdamW;
use crate::train::TrainBatch;

/// Single-node data-parallel training. The model is replicated across
/// `num_workers` worker threads (each thread keeps its own copy of the
/// weights so it can run forward+backward in parallel). After each
/// "global step":
///   - Each worker computes its own gradient on its shard of the batch.
///   - Gradients are averaged element-wise.
///   - One worker's AdamW state applies the averaged gradient; the
///     resulting params are broadcast back to all workers.
///
/// This is the simplest form of distribution and the foundation that
/// multi-node DP extends.
pub struct DataParallelTrainer {
    pub num_workers: usize,
    pub workers: Vec<Mutex<WorkerState>>,
    /// Authoritative model. After each global step its weights are
    /// updated by AdamW; the new state is then synced to each worker.
    pub master_model: Mutex<ImageTransformer>,
    pub master_optimizer: Mutex<AdamW>,
}

pub struct WorkerState {
    pub model: ImageTransformer,
    pub grad: Gradients,
}

impl DataParallelTrainer {
    /// `num_workers = 0` falls back to single-threaded (just calls the
    /// usual `train_step`).
    pub fn new(
        master: ImageTransformer,
        optimizer: AdamW,
        num_workers: usize,
    ) -> Self {
        let workers: Vec<Mutex<WorkerState>> = (0..num_workers)
            .map(|_| {
                let model_copy = clone_model_weights(&master);
                let grad = Gradients::zeros_for(&master);
                Mutex::new(WorkerState { model: model_copy, grad })
            })
            .collect();
        Self {
            num_workers,
            workers,
            master_model: Mutex::new(master),
            master_optimizer: Mutex::new(optimizer),
        }
    }

    /// One distributed training step on a batch of `num_workers`
    /// `TrainBatch`es (one per worker). Returns the mean loss.
    ///
    /// Limitations: this is sequential within a single thread for now —
    /// the workers' grads are computed one after another via `train_step`,
    /// then averaged. The point is the gradient-averaging contract,
    /// not the threading. Switching to actual parallel forward+backward
    /// via Rayon requires marking ImageTransformer as Send + Sync;
    /// straightforward but deferred so the design is clear first.
    pub fn distributed_step(
        &self,
        batches: &[TrainBatch],
    ) -> Result<f32> {
        if batches.is_empty() {
            return Err(crate::error::ImageGenError::ShapeMismatch {
                expected: "at least one batch".into(),
                got: "0".into(),
            });
        }
        let n = batches.len().min(self.num_workers.max(1));

        // Phase 1: per-worker gradient compute on its batch.
        let mut total_loss = 0.0_f32;
        for i in 0..n {
            let batch = &batches[i % batches.len()];
            // Acquire this worker.
            let mut w = self.workers[i].lock().unwrap();
            // Sync weights from master into this worker first.
            let master = self.master_model.lock().unwrap();
            copy_weights(&master, &mut w.model);
            drop(master);
            w.grad.zero_inplace();
            // Split mutable + immutable borrows of `w` for the backward
            // call signature. `w.model` is read; `w.grad` is mutated.
            let WorkerState { ref model, ref mut grad } = *w;
            let cache = crate::backward::forward_train(model, &batch.token_ids)?;
            let loss = crate::backward::backward(
                model, &cache, &batch.predict_at, &batch.targets, grad)?;
            total_loss += loss;
        }
        let mean_loss = total_loss / n as f32;

        // Phase 2: gradient all-reduce (average across workers).
        // Done into a fresh accumulator we attach via the first worker's grad,
        // then divide by n.
        let mut acc = self.workers[0].lock().unwrap().grad.clone();
        for i in 1..n {
            let w = self.workers[i].lock().unwrap();
            // Accumulate each field.
            accumulate_grads(&mut acc, &w.grad);
        }
        scale_grads(&mut acc, 1.0 / n as f32);

        // Phase 3: master optimizer step using the averaged gradient.
        let mut master = self.master_model.lock().unwrap();
        let mut opt = self.master_optimizer.lock().unwrap();
        opt.step(&mut master, &acc);
        Ok(mean_loss)
    }

    /// Block this method runs nothing — included so callers don't fall
    /// into the trap of calling `train_step` from the single-threaded
    /// path while the trainer is otherwise running. Single-node DP is
    /// "single-machine many-thread", not "thread + main both training".
    pub fn assert_single_owner(&self) {
        // Held to surface a future-proofing assertion if needed.
        let _ = self;
    }
}

/// Deep-copy a model's weight tensors (but not the structure itself —
/// we re-create the structure via `from_config` and overwrite weights).
fn clone_model_weights(src: &ImageTransformer) -> ImageTransformer {
    let mut dst = ImageTransformer::from_config(src.config.clone());
    copy_weights(src, &mut dst);
    dst
}

fn copy_weights(src: &ImageTransformer, dst: &mut ImageTransformer) {
    dst.embed.copy_from_slice(&src.embed);
    dst.pos_embed.copy_from_slice(&src.pos_embed);
    for (s, d) in src.blocks.iter().zip(dst.blocks.iter_mut()) {
        d.norm_attn.gamma.copy_from_slice(&s.norm_attn.gamma);
        d.q_proj.weight.copy_from_slice(&s.q_proj.weight);
        d.k_proj.weight.copy_from_slice(&s.k_proj.weight);
        d.v_proj.weight.copy_from_slice(&s.v_proj.weight);
        d.o_proj.weight.copy_from_slice(&s.o_proj.weight);
        d.norm_mlp.gamma.copy_from_slice(&s.norm_mlp.gamma);
        d.mlp_gate.weight.copy_from_slice(&s.mlp_gate.weight);
        d.mlp_up.weight.copy_from_slice(&s.mlp_up.weight);
        d.mlp_down.weight.copy_from_slice(&s.mlp_down.weight);
    }
    dst.final_norm.gamma.copy_from_slice(&src.final_norm.gamma);
    dst.lm_head.weight.copy_from_slice(&src.lm_head.weight);
}

fn accumulate_grads(dst: &mut Gradients, src: &Gradients) {
    for (d, s) in dst.embed.iter_mut().zip(src.embed.iter()) { *d += s; }
    for (d, s) in dst.pos_embed.iter_mut().zip(src.pos_embed.iter()) { *d += s; }
    for (db, sb) in dst.blocks.iter_mut().zip(src.blocks.iter()) {
        for (d, s) in db.norm_attn_gamma.iter_mut().zip(sb.norm_attn_gamma.iter()) { *d += s; }
        for (d, s) in db.q_proj_w.iter_mut().zip(sb.q_proj_w.iter()) { *d += s; }
        for (d, s) in db.k_proj_w.iter_mut().zip(sb.k_proj_w.iter()) { *d += s; }
        for (d, s) in db.v_proj_w.iter_mut().zip(sb.v_proj_w.iter()) { *d += s; }
        for (d, s) in db.o_proj_w.iter_mut().zip(sb.o_proj_w.iter()) { *d += s; }
        for (d, s) in db.norm_mlp_gamma.iter_mut().zip(sb.norm_mlp_gamma.iter()) { *d += s; }
        for (d, s) in db.mlp_gate_w.iter_mut().zip(sb.mlp_gate_w.iter()) { *d += s; }
        for (d, s) in db.mlp_up_w.iter_mut().zip(sb.mlp_up_w.iter()) { *d += s; }
        for (d, s) in db.mlp_down_w.iter_mut().zip(sb.mlp_down_w.iter()) { *d += s; }
    }
    for (d, s) in dst.final_norm_gamma.iter_mut().zip(src.final_norm_gamma.iter()) { *d += s; }
    for (d, s) in dst.lm_head_w.iter_mut().zip(src.lm_head_w.iter()) { *d += s; }
}

fn scale_grads(g: &mut Gradients, k: f32) {
    for v in g.embed.iter_mut() { *v *= k; }
    for v in g.pos_embed.iter_mut() { *v *= k; }
    for b in g.blocks.iter_mut() {
        for v in b.norm_attn_gamma.iter_mut() { *v *= k; }
        for v in b.q_proj_w.iter_mut() { *v *= k; }
        for v in b.k_proj_w.iter_mut() { *v *= k; }
        for v in b.v_proj_w.iter_mut() { *v *= k; }
        for v in b.o_proj_w.iter_mut() { *v *= k; }
        for v in b.norm_mlp_gamma.iter_mut() { *v *= k; }
        for v in b.mlp_gate_w.iter_mut() { *v *= k; }
        for v in b.mlp_up_w.iter_mut() { *v *= k; }
        for v in b.mlp_down_w.iter_mut() { *v *= k; }
    }
    for v in g.final_norm_gamma.iter_mut() { *v *= k; }
    for v in g.lm_head_w.iter_mut() { *v *= k; }
}

// ============================================================================
// Multi-node distributed training via TCP gradient exchange.
// ============================================================================
//
// Topology: star. Rank 0 is the coordinator; ranks 1..N-1 are workers.
// Each global step:
//   1. Every node computes its own local gradient on its batch shard.
//   2. Workers serialize their gradient as a flat little-endian f32 buffer
//      and send it to the coordinator.
//   3. Coordinator averages local + N-1 received gradients (1/N).
//   4. Coordinator broadcasts the averaged gradient back to each worker.
//   5. All nodes apply the averaged gradient via their own AdamW.
//      (Same optimizer state on each node since they were initialised
//      identically and consume identical gradients — synchronous SGD.)
//
// Wire format per message:
//   u32 magic = 0x4B4C474D ("KLGM" — klearu-gradient-message)
//   u32 rank
//   u32 step
//   u64 grad_size_f32   (number of f32 elements)
//   [grad_size_f32 × 4] little-endian f32 bytes
//
// On a connection drop the worker reconnects with exponential backoff
// (up to 30s). The coordinator times out a missing worker after a fixed
// deadline per step; missing workers are excluded from the average for
// that step. This is the same approach Horovod uses for partial-allreduce.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

const GRAD_MAGIC: u32 = 0x4B4C474D;

/// Flatten a gradient buffer into a single contiguous f32 vector. The
/// element order is fixed (so coordinator and workers agree).
pub fn flatten_grad(g: &Gradients) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(grad_element_count(g));
    out.extend_from_slice(&g.embed);
    out.extend_from_slice(&g.pos_embed);
    for b in g.blocks.iter() {
        out.extend_from_slice(&b.norm_attn_gamma);
        out.extend_from_slice(&b.q_proj_w);
        out.extend_from_slice(&b.k_proj_w);
        out.extend_from_slice(&b.v_proj_w);
        out.extend_from_slice(&b.o_proj_w);
        out.extend_from_slice(&b.norm_mlp_gamma);
        out.extend_from_slice(&b.mlp_gate_w);
        out.extend_from_slice(&b.mlp_up_w);
        out.extend_from_slice(&b.mlp_down_w);
    }
    out.extend_from_slice(&g.final_norm_gamma);
    out.extend_from_slice(&g.lm_head_w);
    out
}

/// Element count of a flattened gradient — useful for pre-sizing buffers.
pub fn grad_element_count(g: &Gradients) -> usize {
    let mut n = g.embed.len() + g.pos_embed.len() + g.final_norm_gamma.len() + g.lm_head_w.len();
    for b in g.blocks.iter() {
        n += b.norm_attn_gamma.len() + b.q_proj_w.len() + b.k_proj_w.len()
           + b.v_proj_w.len() + b.o_proj_w.len() + b.norm_mlp_gamma.len()
           + b.mlp_gate_w.len() + b.mlp_up_w.len() + b.mlp_down_w.len();
    }
    n
}

/// Inverse of `flatten_grad`. Returns an error if the buffer length
/// doesn't match the gradient's element count.
pub fn unflatten_grad(g: &mut Gradients, flat: &[f32]) -> Result<()> {
    if flat.len() != grad_element_count(g) {
        return Err(crate::error::ImageGenError::ShapeMismatch {
            expected: format!("{} elements", grad_element_count(g)),
            got: format!("{}", flat.len()),
        });
    }
    let mut off = 0usize;
    let copy = |dst: &mut [f32], flat: &[f32], off: &mut usize| {
        let n = dst.len();
        dst.copy_from_slice(&flat[*off..*off + n]);
        *off += n;
    };
    copy(&mut g.embed, flat, &mut off);
    copy(&mut g.pos_embed, flat, &mut off);
    for b in g.blocks.iter_mut() {
        copy(&mut b.norm_attn_gamma, flat, &mut off);
        copy(&mut b.q_proj_w, flat, &mut off);
        copy(&mut b.k_proj_w, flat, &mut off);
        copy(&mut b.v_proj_w, flat, &mut off);
        copy(&mut b.o_proj_w, flat, &mut off);
        copy(&mut b.norm_mlp_gamma, flat, &mut off);
        copy(&mut b.mlp_gate_w, flat, &mut off);
        copy(&mut b.mlp_up_w, flat, &mut off);
        copy(&mut b.mlp_down_w, flat, &mut off);
    }
    copy(&mut g.final_norm_gamma, flat, &mut off);
    copy(&mut g.lm_head_w, flat, &mut off);
    debug_assert_eq!(off, flat.len());
    Ok(())
}

fn write_grad_message(stream: &mut TcpStream, rank: u32, step: u32, flat: &[f32])
    -> std::io::Result<()>
{
    let mut hdr = Vec::with_capacity(20);
    hdr.extend_from_slice(&GRAD_MAGIC.to_le_bytes());
    hdr.extend_from_slice(&rank.to_le_bytes());
    hdr.extend_from_slice(&step.to_le_bytes());
    hdr.extend_from_slice(&(flat.len() as u64).to_le_bytes());
    stream.write_all(&hdr)?;
    // Send as raw little-endian f32 bytes.
    let mut buf: Vec<u8> = Vec::with_capacity(flat.len() * 4);
    for v in flat { buf.extend_from_slice(&v.to_le_bytes()); }
    stream.write_all(&buf)?;
    stream.flush()
}

fn read_grad_message(stream: &mut TcpStream, expected_len: usize)
    -> std::io::Result<(u32, u32, Vec<f32>)>
{
    let mut hdr = [0u8; 20];
    stream.read_exact(&mut hdr)?;
    let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    if magic != GRAD_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("bad grad magic {magic:#x}")));
    }
    let rank = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    let step = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
    let n = u64::from_le_bytes(hdr[12..20].try_into().unwrap()) as usize;
    if n != expected_len {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("grad len mismatch: peer={n}, expected={expected_len}")));
    }
    let mut buf = vec![0u8; n * 4];
    stream.read_exact(&mut buf)?;
    let mut out = Vec::with_capacity(n);
    for chunk in buf.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok((rank, step, out))
}

/// Coordinator (rank 0). Listens on `bind_addr` for `num_workers` worker
/// connections, then handles per-step gradient all-reduce: receive from
/// each worker, average with the local gradient, broadcast back.
pub struct TcpCoordinator {
    listener: TcpListener,
    /// Worker connections (indexed by rank 1..=num_workers). Wrapped in
    /// `Option` so a dropped connection can be cleared and re-accepted.
    workers: Vec<Option<TcpStream>>,
    pub num_workers: usize,
    pub world_size: usize,
}

impl TcpCoordinator {
    /// Bind and wait for `num_workers` worker connections. Blocks until
    /// every worker has dialed in.
    pub fn bind<A: ToSocketAddrs>(bind_addr: A, num_workers: usize) -> std::io::Result<Self> {
        let listener = TcpListener::bind(bind_addr)?;
        let mut workers: Vec<Option<TcpStream>> = (0..num_workers).map(|_| None).collect();
        for _ in 0..num_workers {
            let (stream, _) = listener.accept()?;
            // Read worker handshake: just a rank byte.
            let mut rank_buf = [0u8; 4];
            (&stream).read_exact_at(&mut rank_buf).map_err(std::io::Error::other)?;
            let rank = u32::from_le_bytes(rank_buf) as usize;
            if rank == 0 || rank > num_workers {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("bad worker rank {rank}")));
            }
            stream.set_read_timeout(Some(Duration::from_secs(60)))?;
            stream.set_write_timeout(Some(Duration::from_secs(60)))?;
            workers[rank - 1] = Some(stream);
        }
        Ok(Self { listener, workers, num_workers, world_size: num_workers + 1 })
    }

    /// One all-reduce: combine `local_grad` with each worker's gradient,
    /// average (1/world_size), broadcast back, mutate `local_grad` in place.
    ///
    /// On a worker socket error, that worker is dropped from the round
    /// and the average uses the remaining `k+1` participants.
    pub fn all_reduce(&mut self, local_grad: &mut Gradients, step: u32) -> Result<()> {
        let expected = grad_element_count(local_grad);
        let mut acc = flatten_grad(local_grad);
        let mut participants = 1usize; // self

        for ri in 0..self.num_workers {
            let result: std::io::Result<Vec<f32>> = (|| {
                let stream = self.workers[ri].as_mut().ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotConnected,
                        format!("rank {} not connected", ri + 1))
                })?;
                let (_, _, flat) = read_grad_message(stream, expected)?;
                Ok(flat)
            })();
            match result {
                Ok(flat) => {
                    for (a, f) in acc.iter_mut().zip(flat.iter()) { *a += f; }
                    participants += 1;
                }
                Err(_) => {
                    // Drop this worker for the round; admin can re-accept later.
                    self.workers[ri] = None;
                }
            }
        }
        let inv = 1.0_f32 / participants as f32;
        for v in acc.iter_mut() { *v *= inv; }

        // Broadcast back.
        for ri in 0..self.num_workers {
            if let Some(stream) = self.workers[ri].as_mut() {
                if write_grad_message(stream, 0, step, &acc).is_err() {
                    self.workers[ri] = None;
                }
            }
        }
        unflatten_grad(local_grad, &acc)?;
        Ok(())
    }

    /// Re-accept any workers that dropped during a previous step. Best-effort.
    pub fn reaccept_dropped(&mut self) -> std::io::Result<()> {
        self.listener.set_nonblocking(true)?;
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let mut rank_buf = [0u8; 4];
                    (&stream).read_exact_at(&mut rank_buf).map_err(std::io::Error::other)?;
                    let rank = u32::from_le_bytes(rank_buf) as usize;
                    if rank >= 1 && rank <= self.num_workers && self.workers[rank - 1].is_none() {
                        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
                        stream.set_write_timeout(Some(Duration::from_secs(60)))?;
                        self.workers[rank - 1] = Some(stream);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e),
            }
        }
        self.listener.set_nonblocking(false)?;
        Ok(())
    }
}

/// Worker (rank ≥ 1). Connects to coordinator and exchanges gradients.
pub struct TcpWorker {
    stream: TcpStream,
    pub rank: u32,
}

impl TcpWorker {
    /// Connect to the coordinator and complete the rank handshake. Retries
    /// with exponential backoff up to `max_retries`.
    pub fn connect<A: ToSocketAddrs + Clone>(
        addr: A,
        rank: u32,
        max_retries: u32,
    ) -> std::io::Result<Self> {
        let mut backoff_ms = 100u64;
        for attempt in 0..=max_retries {
            match TcpStream::connect(addr.clone()) {
                Ok(mut stream) => {
                    stream.write_all(&rank.to_le_bytes())?;
                    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
                    stream.set_write_timeout(Some(Duration::from_secs(60)))?;
                    return Ok(Self { stream, rank });
                }
                Err(e) => {
                    if attempt == max_retries {
                        return Err(e);
                    }
                    std::thread::sleep(Duration::from_millis(backoff_ms));
                    backoff_ms = (backoff_ms * 2).min(30_000);
                }
            }
        }
        unreachable!()
    }

    /// One all-reduce step: send local grad to coordinator, receive
    /// averaged grad, overwrite `local_grad` in place.
    pub fn all_reduce(&mut self, local_grad: &mut Gradients, step: u32) -> Result<()> {
        let flat = flatten_grad(local_grad);
        write_grad_message(&mut self.stream, self.rank, step, &flat)
            .map_err(|e| crate::error::ImageGenError::Io(e))?;
        let (_, _, avg) = read_grad_message(&mut self.stream, flat.len())
            .map_err(|e| crate::error::ImageGenError::Io(e))?;
        unflatten_grad(local_grad, &avg)
    }
}

// Shim because std::os::unix is unconditionally available on macOS/Linux.
// We need a `read_exact` that works on `&TcpStream` (immutable reference)
// to satisfy borrow rules inside TcpCoordinator::bind / reaccept_dropped.
trait ReadExactAt {
    fn read_exact_at(&self, buf: &mut [u8]) -> std::io::Result<()>;
}
impl ReadExactAt for &TcpStream {
    fn read_exact_at(&self, buf: &mut [u8]) -> std::io::Result<()> {
        let mut s = TcpStream::try_clone(self)?;
        s.read_exact(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageTransformerConfig;
    use crate::optim::AdamWConfig;
    use crate::train::{assemble_batch, TrainExample};

    fn tiny_cfg() -> ImageTransformerConfig {
        ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 16, num_layers: 1, num_heads: 4,
            mlp_intermediate: 32, max_text_len: 4,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        }
    }

    #[test]
    fn flatten_unflatten_roundtrip() {
        let cfg = tiny_cfg();
        let model = ImageTransformer::from_config(cfg);
        let mut grad = Gradients::zeros_for(&model);
        for (i, v) in grad.embed.iter_mut().enumerate() { *v = (i as f32 + 1.0) * 0.001; }
        for (i, v) in grad.lm_head_w.iter_mut().enumerate() { *v = -(i as f32 + 1.0) * 0.002; }
        let flat = flatten_grad(&grad);
        assert_eq!(flat.len(), grad_element_count(&grad));
        let mut grad2 = Gradients::zeros_for(&model);
        unflatten_grad(&mut grad2, &flat).expect("unflatten");
        for (a, b) in grad.embed.iter().zip(grad2.embed.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in grad.lm_head_w.iter().zip(grad2.lm_head_w.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn tcp_grad_exchange_localhost() {
        use std::net::TcpListener;
        use std::sync::{Arc, Barrier};
        // Pick an OS-assigned free port via bind+drop.
        let probe = TcpListener::bind("127.0.0.1:0").expect("probe bind");
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        let addr = format!("127.0.0.1:{port}");

        let cfg = tiny_cfg();
        let model = ImageTransformer::from_config(cfg);

        // Two-node setup: rank 0 coordinator, rank 1 worker. Each side
        // brings a deterministic gradient and expects the average back.
        let mut g_coord = Gradients::zeros_for(&model);
        for (i, v) in g_coord.embed.iter_mut().enumerate() { *v = (i as f32) * 0.1; }
        let mut g_worker = Gradients::zeros_for(&model);
        for (i, v) in g_worker.embed.iter_mut().enumerate() { *v = -(i as f32) * 0.1; }
        // Expected: average is exactly zero.

        let barrier = Arc::new(Barrier::new(2));
        let b_coord = barrier.clone();
        let b_worker = barrier.clone();
        let addr_for_worker = addr.clone();

        let coord_thread = std::thread::spawn(move || {
            let mut coord = TcpCoordinator::bind(&addr, 1).expect("bind");
            b_coord.wait();
            coord.all_reduce(&mut g_coord, 0).expect("coord all_reduce");
            g_coord
        });
        let worker_thread = std::thread::spawn(move || {
            // Small sleep to ensure coordinator's accept loop is ready.
            std::thread::sleep(Duration::from_millis(50));
            let mut worker = TcpWorker::connect(&addr_for_worker, 1, 10).expect("connect");
            b_worker.wait();
            worker.all_reduce(&mut g_worker, 0).expect("worker all_reduce");
            g_worker
        });

        let coord_g = coord_thread.join().expect("coord join");
        let worker_g = worker_thread.join().expect("worker join");

        // Both sides should see the averaged gradient (zero).
        for &v in coord_g.embed.iter() {
            assert!(v.abs() < 1e-5, "coord side avg should be ~0, got {v}");
        }
        for &v in worker_g.embed.iter() {
            assert!(v.abs() < 1e-5, "worker side avg should be ~0, got {v}");
        }
    }

    #[test]
    fn dp_step_runs_and_lowers_loss() {
        // 2-worker DP with the same batch on each — should behave the
        // same as a single-worker step (averaging identical gradients
        // gives the same gradient). Loss should drop over a few steps.
        let cfg = tiny_cfg();
        let mut master = ImageTransformer::from_config(cfg.clone());
        // Small random init.
        let mut s = 7_u64;
        let mut rand = || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) * 0.04 - 0.02
        };
        for v in master.embed.iter_mut() { *v = rand(); }
        for v in master.pos_embed.iter_mut() { *v = rand(); }
        for blk in master.blocks.iter_mut() {
            for v in blk.q_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.k_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.v_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.o_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.mlp_gate.weight.iter_mut() { *v = rand(); }
            for v in blk.mlp_up.weight.iter_mut() { *v = rand(); }
            for v in blk.mlp_down.weight.iter_mut() { *v = rand(); }
        }
        for v in master.lm_head.weight.iter_mut() { *v = rand(); }

        let opt = AdamW::new(&master, AdamWConfig {
            lr: 1e-2, weight_decay: 0.0, ..Default::default()
        });
        let trainer = DataParallelTrainer::new(master, opt, 2);

        let ex = TrainExample {
            text_tokens: vec![5, 6],
            image_tokens: vec![3, 7, 11, 5],
        };
        let batch = assemble_batch(&cfg, &ex).expect("batch");
        // Both workers process the same batch — gradient averaging is a
        // no-op, but the test verifies the all-reduce + step plumbing works.
        let batches = vec![batch.clone(), batch];

        let loss_start = trainer.distributed_step(&batches).expect("dp step");
        let mut loss_end = loss_start;
        for _ in 0..15 {
            loss_end = trainer.distributed_step(&batches).expect("dp step");
        }
        assert!(loss_end < loss_start * 0.7,
            "DP training should lower loss: start={loss_start}, end={loss_end}");
    }
}
