//! Cosine similarity of vectors and mean-pooled model token embeddings.

use crate::{LlmError, Result};

fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidConfig(message.into())
}

/// Compute cosine similarity in [-1, 1], accumulating in f64.
/// Empty, unequal, zero-norm, and non-finite vectors are rejected.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.is_empty() || a.len() != b.len() {
        return Err(invalid("Vectors must have equal, nonzero dimensions"));
    }
    let (mut dot, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return Err(invalid("Vectors must contain only finite values"));
        }
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        return Err(invalid(
            "Cosine similarity is undefined for zero-norm vectors",
        ));
    }
    Ok((dot / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0))
}

#[cfg(feature = "native-io")]
mod native {
    use super::*;
    use crate::weight::name_map::{parse_weight_name, WeightTarget};
    use safetensors::{Dtype, SafeTensors};
    use std::path::Path;

    /// Memory-mapped input embeddings. The backing file must not be modified
    /// while this object exists. Only requested rows are converted to f32.
    pub struct TokenEmbeddings {
        mmap: memmap2::Mmap,
        offset: usize,
        dtype: Dtype,
        pub tensor_name: String,
        vocab_size: usize,
        hidden_size: usize,
    }

    impl TokenEmbeddings {
        /// Open a safetensors file or a model directory (including sharded models).
        /// Auto-detects the repo's supported embedding names; use `tensor_name`
        /// to select another architecture's [vocabulary, dimension] tensor.
        pub fn open(path: &Path, tensor_name: Option<&str>) -> Result<Self> {
            let mut paths = Vec::new();
            if path.is_dir() {
                for entry in std::fs::read_dir(path)? {
                    let path = entry?.path();
                    if path.extension().is_some_and(|e| e == "safetensors") {
                        paths.push(path);
                    }
                }
                paths.sort();
            } else {
                paths.push(path.to_path_buf());
            }
            let mut found = None;
            for path in paths {
                let file = std::fs::File::open(&path)?;
                // SAFETY: read-only mapping; callers must keep the file unmodified.
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                let (header_len, metadata) = SafeTensors::read_metadata(&mmap)
                    .map_err(|e| LlmError::SafeTensors(format!("{}: {e}", path.display())))?;
                let candidates: Vec<_> = metadata
                    .tensors()
                    .into_iter()
                    .filter(|(name, _)| {
                        tensor_name.map_or_else(
                            || parse_weight_name(&name) == Some(WeightTarget::Embedding),
                            |requested| requested == name,
                        )
                    })
                    .collect();
                if candidates.len() > 1 || (!candidates.is_empty() && found.is_some()) {
                    return Err(invalid(
                        "Multiple embedding tensors found; select a file and --tensor explicitly",
                    ));
                }
                if let Some((name, info)) = candidates.into_iter().next() {
                    if info.shape.len() != 2 || info.shape.contains(&0) {
                        return Err(invalid(format!(
                            "{name} must have nonempty shape [vocab_size, hidden_size]"
                        )));
                    }
                    if !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                        return Err(LlmError::UnsupportedDtype(format!("{:?}", info.dtype)));
                    }
                    found = Some(Self {
                        mmap,
                        offset: 8 + header_len + info.data_offsets.0,
                        dtype: info.dtype,
                        tensor_name: name,
                        vocab_size: info.shape[0],
                        hidden_size: info.shape[1],
                    });
                }
            }
            found.ok_or_else(|| {
                LlmError::MissingWeight(
                    tensor_name
                        .unwrap_or("input embeddings (use --tensor to specify a name)")
                        .into(),
                )
            })
        }

        pub fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        pub fn hidden_size(&self) -> usize {
            self.hidden_size
        }

        /// Arithmetic mean of input embeddings. Repeated IDs retain their weight;
        /// order does not affect the result. This does not run a transformer.
        pub fn mean_embedding(&self, ids: &[u32]) -> Result<Vec<f32>> {
            if ids.is_empty() {
                return Err(invalid("Token combination must not be empty"));
            }
            let mut sum = vec![0.0_f64; self.hidden_size];
            let width = self.dtype.size();
            for &id in ids {
                if id as usize >= self.vocab_size {
                    return Err(invalid(format!(
                        "Token ID {id} is outside vocabulary of size {}",
                        self.vocab_size
                    )));
                }
                let start = self.offset + id as usize * self.hidden_size * width;
                let row = &self.mmap[start..start + self.hidden_size * width];
                for (total, bytes) in sum.iter_mut().zip(row.chunks_exact(width)) {
                    let value = match self.dtype {
                        Dtype::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
                        Dtype::F16 => {
                            half::f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()))
                                .to_f32()
                        }
                        Dtype::BF16 => {
                            half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()))
                                .to_f32()
                        }
                        _ => unreachable!(),
                    };
                    if !value.is_finite() {
                        return Err(invalid(format!("Token {id} has a non-finite embedding")));
                    }
                    *total += f64::from(value);
                }
            }
            Ok(sum
                .into_iter()
                .map(|v| (v / ids.len() as f64) as f32)
                .collect())
        }
    }
}

#[cfg(feature = "native-io")]
pub use native::TokenEmbeddings;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_geometry_and_extremes() {
        for (b, expected) in [([1.0, 0.0], 1.0), ([0.0, 1.0], 0.0), ([-1.0, 0.0], -1.0)] {
            assert!((cosine_similarity(&[1.0, 0.0], &b).unwrap() - expected).abs() < 1e-12);
        }
        assert!(
            (cosine_similarity(&[1.0, 1.0], &[3.0, 0.0]).unwrap() - 1.0 / 2.0_f64.sqrt()).abs()
                < 1e-12
        );
        for x in [f32::MAX, f32::MIN_POSITIVE] {
            assert!((cosine_similarity(&[x, x], &[x, x]).unwrap() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn invalid_vectors() {
        for (a, b) in [
            (vec![], vec![]),
            (vec![1.0], vec![1.0, 2.0]),
            (vec![0.0], vec![1.0]),
            (vec![f32::NAN], vec![1.0]),
            (vec![1.0], vec![f32::INFINITY]),
        ] {
            assert!(cosine_similarity(&a, &b).is_err());
        }
    }
}
