//! Load tensors from safetensors files by name.
//!
//! Two access paths:
//!   - [`ComponentTensors::get_f32`] — coerces any supported dtype to f32.
//!     The CPU reference path, and the only option for math done on the host.
//!   - [`ComponentTensors::get_bf16`] — yields raw little-endian bf16 bytes
//!     ready to `memcpy` into an `MTLBuffer`. For a bf16-native checkpoint
//!     this borrows straight out of the mmap: no conversion, no f32
//!     temporary, no allocation.
//!
//! The GPU pipeline stores **bf16** (see `metal_backend`, where the historic
//! `GpuDtype::F16` name denotes bf16 storage). Routing a bf16 checkpoint
//! through `get_f32` and back is bit-exact but pointlessly expensive — it
//! materialises a 2x-sized f32 temporary per tensor. `get_bf16` is the path
//! that avoids it.

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
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                files.push(p);
            }
        }
        Self::open_files(&files)
    }

    /// Open one `.safetensors` file — a single-file checkpoint.
    pub fn open_file(path: &Path) -> Result<Self> {
        Self::open_files(&[path.to_path_buf()])
    }

    fn open_files(files: &[std::path::PathBuf]) -> Result<Self> {
        let mut mmaps = Vec::new();
        let mut tensors: HashMap<String, TensorRef> = HashMap::new();
        for p in files {
            let file = std::fs::File::open(p)?;
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
        bytes_to_f32(bytes, r.dtype, r.shape.iter().product())
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
        Dtype::F8_E4M3 => {
            // OFP8 E4M3 (1 sign / 4 exponent / 3 mantissa bits, exponent
            // bias 7). Used by quantized Flux releases (flux1-dev-fp8.…)
            // to halve checkpoint size. We dequantize to f32 at load time.
            if bytes.len() != expected_count {
                return Err(DiffusionError::ShapeMismatch {
                    expected: format!("{expected_count} bytes (fp8 E4M3)"),
                    got: format!("{}", bytes.len()),
                });
            }
            for &b in bytes {
                out.push(f8_e4m3_to_f32(b));
            }
        }
        Dtype::F8_E5M2 => {
            // OFP8 E5M2 (1/5/2, exponent bias 15). Less common in Flux
            // checkpoints but ships with some quantized variants.
            if bytes.len() != expected_count {
                return Err(DiffusionError::ShapeMismatch {
                    expected: format!("{expected_count} bytes (fp8 E5M2)"),
                    got: format!("{}", bytes.len()),
                });
            }
            for &b in bytes {
                out.push(f8_e5m2_to_f32(b));
            }
        }
        other => {
            return Err(DiffusionError::Unsupported(format!("dtype {other:?}")));
        }
    }
    Ok(out)
}

/// Decode one OFP8 E4M3 byte to f32. Layout: bit 7 = sign, bits 3-6 =
/// exponent (bias 7), bits 0-2 = mantissa. Special: e=0xF & m=0x7 → NaN
/// (E4M3 has no infinities). Subnormals: e=0 ⇒ value = (-1)^s · 2^-6 · m/8.
fn f8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 0x1;
    let exp = (b >> 3) & 0xF;
    let mant = b & 0x7;
    let s: f32 = if sign == 1 { -1.0 } else { 1.0 };
    if exp == 0 {
        // Zero or subnormal. value = s · 2^-6 · (mant / 8)
        if mant == 0 { return s * 0.0; }
        return s * (mant as f32) * (1.0 / 8.0) * (1.0 / 64.0); // 2^-6 = 1/64
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN; // E4M3's only NaN encoding
    }
    // Normal: value = s · 2^(exp - 7) · (1 + mant/8)
    let exponent = (exp as i32) - 7;
    let mantissa = 1.0 + (mant as f32) * (1.0 / 8.0);
    let pow = if exponent >= 0 {
        (1u64 << exponent) as f32
    } else {
        1.0 / (1u64 << (-exponent)) as f32
    };
    s * pow * mantissa
}

/// Decode one OFP8 E5M2 byte to f32. Layout: 1/5/2, exponent bias 15.
/// E5M2 *does* have infinities (e=0x1F & m=0) and NaNs (e=0x1F & m≠0).
fn f8_e5m2_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 0x1;
    let exp = (b >> 2) & 0x1F;
    let mant = b & 0x3;
    let s: f32 = if sign == 1 { -1.0 } else { 1.0 };
    if exp == 0 {
        if mant == 0 { return s * 0.0; }
        // Subnormal: s · 2^-14 · (mant / 4)
        return s * (mant as f32) * 0.25 * (1.0 / 16384.0);
    }
    if exp == 0x1F {
        return if mant == 0 { s * f32::INFINITY } else { f32::NAN };
    }
    let exponent = (exp as i32) - 15;
    let mantissa = 1.0 + (mant as f32) * 0.25;
    let pow = if exponent >= 0 {
        (1u64 << exponent) as f32
    } else {
        1.0 / (1u64 << (-exponent)) as f32
    };
    s * pow * mantissa
}

/// Convenience: try several names in order, return the first that exists.
/// Useful for cross-version naming: e.g., `proj_in.weight` vs `proj_in.weight_g`.
pub fn first_present<'a>(
    component: &'a ComponentTensors,
    names: &[&'a str],
) -> Option<&'a str> {
    names.iter().copied().find(|n| component.has(n))
}

// ---------------------------------------------------------------------------
// bf16 access path
// ---------------------------------------------------------------------------

/// How faithfully a source dtype survives the trip into bf16 storage.
///
/// The GPU pipeline stores bf16 unconditionally, so every checkpoint dtype
/// lands there eventually. What differs is the cost and the precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16Fidelity {
    /// Source is already bf16 — the bytes are the destination format.
    /// Zero conversion, zero allocation, borrowed straight from the mmap.
    ZeroCopy,
    /// Every finite source value is exactly representable in bf16. True of
    /// both OFP8 formats: E4M3 (4 exponent, 3 mantissa) and E5M2 (5, 2) fit
    /// inside bf16's 8 exponent / 7 mantissa bits with room to spare, and
    /// their subnormals are ordinary normals at bf16's exponent range.
    Exact,
    /// f32 → bf16 truncates 16 mantissa bits, round-to-nearest-even. This is
    /// the intended, unavoidable cost of bf16 storage — not a defect.
    Rounded,
    /// **Precision is lost beyond the bf16 storage cost.** IEEE f16 carries 10
    /// mantissa bits against bf16's 7, so three bits go missing that a f16
    /// compute path would have kept. Relevant for fp16-native checkpoints —
    /// SD 1.5 and SDXL commonly ship fp16 — where it is a real (if usually
    /// tolerable) quality cost, not just a re-encoding.
    Lossy,
}

impl Bf16Fidelity {
    /// True when converting to bf16 discards information the source held
    /// *and* a different storage choice could have kept it.
    pub fn loses_precision(self) -> bool {
        matches!(self, Bf16Fidelity::Lossy)
    }
}

/// Classify how `dtype` maps onto bf16 storage. `None` for dtypes we cannot
/// decode at all.
pub fn bf16_fidelity(dtype: Dtype) -> Option<Bf16Fidelity> {
    match dtype {
        Dtype::BF16 => Some(Bf16Fidelity::ZeroCopy),
        Dtype::F8_E4M3 | Dtype::F8_E5M2 => Some(Bf16Fidelity::Exact),
        Dtype::F32 => Some(Bf16Fidelity::Rounded),
        Dtype::F16 => Some(Bf16Fidelity::Lossy),
        _ => None,
    }
}

impl ComponentTensors {
    /// The on-disk dtype of a tensor, without touching its data.
    pub fn dtype(&self, name: &str) -> Option<Dtype> {
        self.tensors.get(name).map(|r| r.dtype)
    }

    /// Element count of a tensor, without touching its data.
    pub fn numel(&self, name: &str) -> Option<usize> {
        self.tensors.get(name).map(|r| r.shape.iter().product())
    }

    /// How this tensor will fare when stored as bf16. See [`Bf16Fidelity`].
    pub fn bf16_fidelity_of(&self, name: &str) -> Option<Bf16Fidelity> {
        self.dtype(name).and_then(bf16_fidelity)
    }

    /// Get a tensor as raw **little-endian bf16 bytes**, ready to `memcpy`
    /// into an `MTLBuffer` (`metal_backend::upload_bf16_bytes`).
    ///
    /// A bf16 source returns [`std::borrow::Cow::Borrowed`] — a view into the
    /// mmap with no conversion and no allocation. Every other dtype is decoded
    /// through f32 and re-encoded, returning `Cow::Owned`.
    ///
    /// The returned bytes are deliberately *not* typed as `[bf16]`: safetensors
    /// mmap payloads are not guaranteed to be 2-byte aligned, so they must be
    /// copied rather than reinterpreted in place. Callers memcpy; they never
    /// dereference these as `bf16`.
    pub fn get_bf16(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>> {
        let r = self.tensors.get(name)
            .ok_or_else(|| DiffusionError::MissingTensor(name.to_string()))?;
        let bytes = self.bytes_of(r);
        let count: usize = r.shape.iter().product();

        let expect = |per_elem: usize, label: &str| -> Result<()> {
            if bytes.len() != count * per_elem {
                return Err(DiffusionError::ShapeMismatch {
                    expected: format!("{} bytes ({label})", count * per_elem),
                    got: format!("{}", bytes.len()),
                });
            }
            Ok(())
        };

        match r.dtype {
            // Already the destination format. safetensors is little-endian and
            // every platform we build for is little-endian (asserted below), so
            // the on-disk bytes are the in-memory bytes.
            Dtype::BF16 => {
                expect(2, "bf16")?;
                Ok(std::borrow::Cow::Borrowed(bytes))
            }
            Dtype::F32 => {
                expect(4, "f32")?;
                // Converted in blocks through `convert_from_f32_slice`, which
                // is SIMD (NEON on Apple silicon). A scalar per-element loop
                // costs ~30s on the 5B DiT. The block size keeps the f32
                // staging bounded instead of materialising the whole tensor.
                //
                // Converting into a `Vec<bf16>` rather than straight into a
                // byte buffer keeps the destination naturally aligned —
                // `Vec<u8>` only guarantees alignment 1, so writing bf16
                // through it would rely on the allocator.
                const BLOCK: usize = 16 * 1024;
                use half::slice::HalfFloatSliceExt;
                let mut bf = vec![half::bf16::ZERO; count];
                let mut src: Vec<f32> = Vec::with_capacity(BLOCK);
                for (block, dst) in
                    bytes.chunks(BLOCK * 4).zip(bf.chunks_mut(BLOCK))
                {
                    src.clear();
                    src.extend(
                        block.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                    );
                    dst.convert_from_f32_slice(&src);
                }
                // bf16's in-memory form is its little-endian byte form on every
                // target we build for (asserted below), so this is one memcpy.
                let raw = unsafe {
                    std::slice::from_raw_parts(bf.as_ptr() as *const u8, count * 2)
                };
                Ok(std::borrow::Cow::Owned(raw.to_vec()))
            }
            Dtype::F16 => {
                expect(2, "f16")?;
                let mut out = Vec::with_capacity(count * 2);
                for chunk in bytes.chunks_exact(2) {
                    let v = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                    out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                }
                Ok(std::borrow::Cow::Owned(out))
            }
            Dtype::F8_E4M3 | Dtype::F8_E5M2 => {
                expect(1, "fp8")?;
                let decode = if r.dtype == Dtype::F8_E4M3 { f8_e4m3_to_f32 } else { f8_e5m2_to_f32 };
                let mut out = Vec::with_capacity(count * 2);
                for &b in bytes {
                    out.extend_from_slice(&half::bf16::from_f32(decode(b)).to_le_bytes());
                }
                Ok(std::borrow::Cow::Owned(out))
            }
            other => Err(DiffusionError::Unsupported(format!("dtype {other:?} → bf16"))),
        }
    }
}

// `get_bf16` hands out on-disk bytes verbatim for bf16 sources. safetensors
// stores little-endian; reinterpreting those bytes as native bf16 is only
// correct on a little-endian target.
#[cfg(target_endian = "big")]
compile_error!("klearu-diffusion assumes a little-endian target: \
ComponentTensors::get_bf16 returns safetensors' little-endian payload as \
native bf16 bytes.");

#[cfg(test)]
mod bf16_tests {
    use super::*;

    /// Build a `ComponentTensors` backed entirely by owned bytes.
    fn owned(name: &str, dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) -> ComponentTensors {
        let mut tensors = HashMap::new();
        tensors.insert(name.to_string(), TensorRef {
            mmap_index: OWNED_BACKING,
            offset: 0,
            end: data.len(),
            dtype,
            shape,
        });
        ComponentTensors { mmaps: Vec::new(), owned: data, tensors }
    }

    fn as_bf16(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    #[test]
    fn bf16_source_is_borrowed_not_copied() {
        let vals = [1.0_f32, -2.5, 1e30, -1e-30];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        let c = owned("w", Dtype::BF16, vec![4], data);
        let got = c.get_bf16("w").unwrap();
        assert!(matches!(got, std::borrow::Cow::Borrowed(_)), "bf16 must not allocate");
        assert_eq!(c.bf16_fidelity_of("w"), Some(Bf16Fidelity::ZeroCopy));
        // Bytes round-trip to the same values get_f32 would have produced.
        assert_eq!(as_bf16(&got), c.get_f32("w").unwrap());
    }

    #[test]
    fn f32_source_rounds_to_nearest_even() {
        // 1.0 + 2^-9 sits exactly halfway between two bf16 neighbours
        // (bf16 has 7 explicit mantissa bits, so its step at 1.0 is 2^-7).
        // Round-to-nearest-even must pick the even mantissa, i.e. 1.0.
        // 3.0e38 is inside bf16's range (max ~3.3895e38) but ~4.6e33x above
        // IEEE f16's 65504 ceiling.
        let vals = [1.0_f32, 1.0 + 2.0_f32.powi(-8), 3.0e38, -0.0];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let c = owned("w", Dtype::F32, vec![4], data);
        assert_eq!(c.bf16_fidelity_of("w"), Some(Bf16Fidelity::Rounded));
        let got = as_bf16(&c.get_bf16("w").unwrap());
        for (g, v) in got.iter().zip(vals) {
            assert_eq!(*g, half::bf16::from_f32(v).to_f32());
        }
        // Large f32 stays finite — this is the whole reason storage is bf16
        // rather than IEEE f16, which would have overflowed to Inf at 65504.
        assert!(got[2].is_finite(), "bf16 must hold f32-range magnitudes");
        assert!(got[2] > 2.9e38);
        assert!(half::f16::from_f32(vals[2]).is_infinite(), "f16 would overflow here");
    }

    #[test]
    fn f16_source_is_flagged_lossy_and_actually_loses_bits() {
        // 1.0 + 2^-10 is representable in f16 (10 mantissa bits) but not in
        // bf16 (7), so the value must change. That's the cost the
        // `Lossy` classification exists to advertise.
        let v = 1.0_f32 + 2.0_f32.powi(-10);
        let h = half::f16::from_f32(v);
        assert_eq!(h.to_f32(), v, "value should be exact in f16");
        let c = owned("w", Dtype::F16, vec![1], h.to_le_bytes().to_vec());
        assert_eq!(c.bf16_fidelity_of("w"), Some(Bf16Fidelity::Lossy));
        assert!(Bf16Fidelity::Lossy.loses_precision());
        let got = as_bf16(&c.get_bf16("w").unwrap());
        assert_ne!(got[0], v, "f16 → bf16 must drop mantissa bits here");
        assert_eq!(got[0], 1.0);
    }

    #[test]
    fn fp8_sources_are_exact_in_bf16() {
        // Every one of the 256 E4M3 and E5M2 encodings must survive the trip
        // into bf16 unchanged (NaN compared as NaN).
        for dtype in [Dtype::F8_E4M3, Dtype::F8_E5M2] {
            assert_eq!(bf16_fidelity(dtype), Some(Bf16Fidelity::Exact));
            let data: Vec<u8> = (0u8..=255).collect();
            let c = owned("w", dtype, vec![256], data);
            let via_f32 = c.get_f32("w").unwrap();
            let via_bf16 = as_bf16(&c.get_bf16("w").unwrap());
            for (i, (a, b)) in via_f32.iter().zip(&via_bf16).enumerate() {
                if a.is_nan() {
                    assert!(b.is_nan(), "{dtype:?} byte {i}: NaN became {b}");
                } else {
                    assert_eq!(a, b, "{dtype:?} byte {i} not exact in bf16");
                }
            }
        }
    }

    #[test]
    fn bf16_and_f32_paths_agree_for_every_supported_dtype() {
        // get_bf16 must be exactly `get_f32` followed by an f32→bf16 round,
        // for all dtypes. This is the invariant that lets call sites switch
        // paths without changing numerics.
        let cases: Vec<(Dtype, Vec<u8>, usize)> = vec![
            (Dtype::F32, [1.5_f32, -7.25, 1e-8].iter().flat_map(|v| v.to_le_bytes()).collect(), 3),
            (Dtype::F16, [1.5_f32, -7.25, 1e-8].iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(), 3),
            (Dtype::BF16, [1.5_f32, -7.25, 1e-8].iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(), 3),
            (Dtype::F8_E4M3, vec![0x38, 0xC0, 0x7E], 3),
            (Dtype::F8_E5M2, vec![0x3C, 0xC0, 0x7B], 3),
        ];
        for (dtype, data, n) in cases {
            let c = owned("w", dtype, vec![n], data);
            let want: Vec<f32> = c.get_f32("w").unwrap().iter()
                .map(|v| half::bf16::from_f32(*v).to_f32()).collect();
            let got = as_bf16(&c.get_bf16("w").unwrap());
            assert_eq!(got, want, "{dtype:?}");
        }
    }

    #[test]
    fn wrong_byte_count_is_an_error_not_a_panic() {
        // Claim 4 elements, supply 3 f32s.
        let data: Vec<u8> = [1.0_f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let c = owned("w", Dtype::F32, vec![4], data);
        assert!(c.get_bf16("w").is_err());
    }

    #[test]
    fn missing_and_unsupported_are_reported() {
        let c = owned("w", Dtype::I32, vec![1], vec![0; 4]);
        assert!(matches!(c.get_bf16("nope"), Err(DiffusionError::MissingTensor(_))));
        assert!(matches!(c.get_bf16("w"), Err(DiffusionError::Unsupported(_))));
        assert_eq!(c.bf16_fidelity_of("w"), None);
        assert_eq!(c.dtype("w"), Some(Dtype::I32));
        assert_eq!(c.numel("w"), Some(1));
    }
}
