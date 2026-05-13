//! Weight loading from SD checkpoint files (Diffusers safetensors format).

pub mod flux_checkpoint;
pub mod inventory;
pub mod load_helpers;
pub mod loader;
pub mod single_file;

pub use flux_checkpoint::{
    FluxCheckpoint, FluxPaths, FluxVariant,
    component_with_prefix, detect_transformer_prefix, detect_variant_single_file,
    discover_flux_paths, transformer_component_from_loader, vae_component_from_loader,
};
pub use inventory::{ComponentStats, TensorInfo, inventory_checkpoint, summarise};
pub use load_helpers::{
    load_conv2d, load_embedding, load_group_norm, load_layer_norm, load_linear,
};
pub use loader::{ComponentTensors, TensorRef, first_present};
pub use single_file::{SDFormat, SingleFileLoader};
