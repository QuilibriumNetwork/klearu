//! Walk an SD checkpoint directory and inventory every safetensors file's
//! tensors. Useful for: validating a downloaded checkpoint, building
//! name → expected-shape maps, and surfacing surprises (mismatched dtypes,
//! unexpected components, missing files).

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub component: String, // "unet", "vae", "text_encoder", ...
    pub file: PathBuf,
    pub name: String,
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
}

/// Enumerate all tensors across all `.safetensors` files in a checkpoint
/// directory's known components. Returns a flat list with component
/// labels so callers can group / filter.
pub fn inventory_checkpoint(root: &Path) -> crate::Result<Vec<TensorInfo>> {
    let components = ["unet", "vae", "text_encoder", "text_encoder_2"]; // text_encoder_2 only on SDXL
    let mut out = Vec::new();
    for c in components {
        let dir = root.join(c);
        if !dir.is_dir() { continue; }
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                inventory_file(&p, c, &mut out)?;
            }
        }
    }
    Ok(out)
}

fn inventory_file(path: &Path, component: &str, out: &mut Vec<TensorInfo>) -> crate::Result<()> {
    use memmap2::Mmap;
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = safetensors::SafeTensors::deserialize(&mmap)?;
    for name in st.names() {
        let tensor = st.tensor(name)?;
        out.push(TensorInfo {
            component: component.to_string(),
            file: path.to_path_buf(),
            name: name.to_string(),
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

/// Group inventory results by component and summarise.
pub fn summarise(infos: &[TensorInfo]) -> std::collections::BTreeMap<String, ComponentStats> {
    let mut m: std::collections::BTreeMap<String, ComponentStats> = std::collections::BTreeMap::new();
    for info in infos {
        let s = m.entry(info.component.clone()).or_default();
        s.tensor_count += 1;
        let n: usize = info.shape.iter().product();
        s.total_elements += n;
        let bytes_per_elem = match info.dtype {
            safetensors::Dtype::F32 => 4,
            safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
            safetensors::Dtype::F64 => 8,
            safetensors::Dtype::I8 | safetensors::Dtype::U8 | safetensors::Dtype::BOOL => 1,
            safetensors::Dtype::I16 | safetensors::Dtype::U16 => 2,
            safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
            safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
            _ => 4,
        };
        s.total_bytes += n * bytes_per_elem;
    }
    m
}

#[derive(Debug, Default, Clone)]
pub struct ComponentStats {
    pub tensor_count: usize,
    pub total_elements: usize,
    pub total_bytes: usize,
}
