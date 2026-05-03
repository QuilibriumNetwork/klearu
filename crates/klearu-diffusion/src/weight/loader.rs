//! Load f32 tensors from safetensors files by name. Coerces f16/bf16 to f32.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::error::{DiffusionError, Result};

/// Sentinel for `TensorRef::mmap_index` indicating the bytes live in
/// `ComponentTensors::owned` rather than in any mmap. Used by single-file
/// synthesised tensors (CLIP-G QKV splits, etc.).
pub const OWNED_BACKING: usize = usize::MAX;

/// Lazily-loaded view of one or more safetensors files in a checkpoint
/// directory. Holds mmap handles and per-tensor metadata; `get` fetches
/// (and converts to f32) on demand.
///
/// Tensors can be backed by:
///   - A real mmap (component loaded from a folder; or single-file source)
///   - An owned `Vec<u8>` (synthesised tensors from single-file mode, e.g.
///     OpenCLIP CLIP-G's split QKV weights). Synth tensors set
///     `TensorRef::mmap_index = OWNED_BACKING` and read from `owned` at
///     `[offset, end)`.
pub struct ComponentTensors {
    pub mmaps: Vec<Mmap>,
    /// Owned in-memory bytes. Used for tensor data that has to be
    /// transformed (split, reshaped, dtype-converted) before exposing.
    pub owned: Vec<u8>,
    /// Maps tensor name → (mmap_index, offset, end, dtype, shape).
    /// Stored manually because `SafeTensors` doesn't borrow into Mmap.
    pub tensors: HashMap<String, TensorRef>,
}

#[derive(Clone)]
pub struct TensorRef {
    pub mmap_index: usize,
    pub offset: usize,
    pub end: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl ComponentTensors {
    /// Construct a virtual ComponentTensors that views tensors from a
    /// SingleFileLoader. Tensors are kept as references into the loader's
    /// mmap (no copies) — we don't take ownership of the mmap because the
    /// loader outlives the component. Instead we point through to the
    /// loader's bytes via a snapshot of the mmap data.
    ///
    /// Trade-off: we can't share the loader's `Mmap` directly because
    /// `Mmap` isn't `Clone`. So we copy the *referenced bytes only* (not
    /// the entire file) into our owned buffer at construction time. For
    /// CLIP-G's ~700 MB this copies once at component creation, then all
    /// later synth appends are O(synth bytes) into the same Vec.
    pub fn from_single_mmap_borrow(
        src: &crate::weight::single_file::SingleFileLoader,
        translated: std::collections::HashMap<String, TensorRef>,
    ) -> Self {
        let total_bytes: usize = translated.values()
            .map(|r| r.end - r.offset)
            .sum();
        let mut owned: Vec<u8> = Vec::with_capacity(total_bytes);
        let mut new_translated: std::collections::HashMap<String, TensorRef> =
            std::collections::HashMap::new();
        for (name, r) in &translated {
            let bytes = &src.mmap[r.offset..r.end];
            let new_offset = owned.len();
            owned.extend_from_slice(bytes);
            new_translated.insert(name.clone(), TensorRef {
                mmap_index: OWNED_BACKING,
                offset: new_offset,
                end: new_offset + bytes.len(),
                dtype: r.dtype,
                shape: r.shape.clone(),
            });
        }
        Self {
            mmaps: Vec::new(),
            owned,
            tensors: new_translated,
        }
    }

    /// Append an in-memory synthesised tensor (e.g. one slice of a CLIP-G
    /// `attn.in_proj_weight`) into the owned buffer. Amortised O(data.len)
    /// per call — no filesystem traffic, no rewrites of existing data.
    /// Vec realloc may move existing bytes, but offsets stay valid.
    pub fn append_synth_tensor(
        &mut self,
        name: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        data: &[u8],
    ) -> Result<()> {
        let new_offset = self.owned.len();
        self.owned.extend_from_slice(data);
        self.tensors.insert(name.to_string(), TensorRef {
            mmap_index: OWNED_BACKING,
            offset: new_offset,
            end: new_offset + data.len(),
            dtype,
            shape,
        });
        Ok(())
    }

    /// Resolve a tensor's bytes regardless of backing.
    fn bytes_of<'a>(&'a self, r: &TensorRef) -> &'a [u8] {
        if r.mmap_index == OWNED_BACKING {
            &self.owned[r.offset..r.end]
        } else {
            &self.mmaps[r.mmap_index][r.offset..r.end]
        }
    }

    /// Open every `.safetensors` file in `dir` and build the name → tensor map.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut mmaps = Vec::new();
        let mut tensors: HashMap<String, TensorRef> = HashMap::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("safetensors") {
                continue;
            }
            let file = std::fs::File::open(&p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let mmap_index = mmaps.len();
            let st = SafeTensors::deserialize(&mmap)?;
            for name in st.names() {
                let tensor = st.tensor(name)?;
                let data = tensor.data();
                // tensor.data() returns a slice into mmap. Compute its offset
                // by pointer arithmetic against the mmap base.
                let base = mmap.as_ptr() as usize;
                let dptr = data.as_ptr() as usize;
                let offset = dptr - base;
                let end = offset + data.len();
                tensors.insert(name.to_string(), TensorRef {
                    mmap_index,
                    offset,
                    end,
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                });
            }
            mmaps.push(mmap);
        }
        Ok(Self { mmaps, owned: Vec::new(), tensors })
    }

    /// Get a tensor by name as f32. Allocates a fresh Vec.
    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>> {
        let r = self.tensors.get(name)
            .ok_or_else(|| DiffusionError::MissingTensor(name.to_string()))?;
        let bytes = self.bytes_of(r);
        Ok(bytes_to_f32(bytes, r.dtype, r.shape.iter().product())?)
    }

    /// Get a tensor's shape without loading the data.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|r| r.shape.as_slice())
    }

    /// True iff the named tensor exists.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

fn bytes_to_f32(bytes: &[u8], dtype: Dtype, expected_count: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(expected_count);
    match dtype {
        Dtype::F32 => {
            if bytes.len() != expected_count * 4 {
                return Err(DiffusionError::ShapeMismatch {
                    expected: format!("{} bytes (f32)", expected_count * 4),
                    got: format!("{}", bytes.len()),
                });
            }
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Dtype::F16 => {
            if bytes.len() != expected_count * 2 {
                return Err(DiffusionError::ShapeMismatch {
                    expected: format!("{} bytes (f16)", expected_count * 2),
                    got: format!("{}", bytes.len()),
                });
            }
            for chunk in bytes.chunks_exact(2) {
                out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        Dtype::BF16 => {
            if bytes.len() != expected_count * 2 {
                return Err(DiffusionError::ShapeMismatch {
                    expected: format!("{} bytes (bf16)", expected_count * 2),
                    got: format!("{}", bytes.len()),
                });
            }
            for chunk in bytes.chunks_exact(2) {
                out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        other => {
            return Err(DiffusionError::Unsupported(format!("dtype {other:?}")));
        }
    }
    Ok(out)
}

/// Convenience: try several names in order, return the first that exists.
/// Useful for cross-version naming: e.g., `proj_in.weight` vs `proj_in.weight_g`.
pub fn first_present<'a>(
    component: &'a ComponentTensors,
    names: &[&'a str],
) -> Option<&'a str> {
    names.iter().copied().find(|n| component.has(n))
}
