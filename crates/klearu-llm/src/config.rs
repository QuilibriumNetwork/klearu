use serde::{Deserialize, Serialize};

use crate::error::{LlmError, Result};

/// Layer type for hybrid attention models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Standard full attention (LLaMA-style or Qwen3.5 gated full attention).
    FullAttention,
    /// GatedDeltaNet linear attention (Qwen3.5).
    LinearAttention,
}

/// LLaMA-compatible model configuration, extended for Qwen3.5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(alias = "num_attention_heads")]
    pub num_heads: usize,
    #[serde(alias = "num_key_value_heads")]
    pub num_kv_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub intermediate_size: usize,
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(alias = "max_position_embeddings", default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // --- Qwen3.5-specific fields ---

    /// Model type identifier (e.g., "qwen3_5_text", "llama").
    #[serde(default)]
    pub model_type: Option<String>,

    /// Per-layer attention type: "linear_attention" or "full_attention".
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    /// Interval between full attention layers (default 4).
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,

    /// Number of key heads for linear attention layers.
    #[serde(default)]
    pub linear_num_key_heads: usize,

    /// Number of value heads for linear attention layers.
    #[serde(default)]
    pub linear_num_value_heads: usize,

    /// Key head dimension for linear attention layers.
    #[serde(default)]
    pub linear_key_head_dim: usize,

    /// Value head dimension for linear attention layers.
    #[serde(default)]
    pub linear_value_head_dim: usize,

    /// Depthwise conv kernel size for linear attention (default 4).
    #[serde(default = "default_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,

    /// Whether full-attention layers use a sigmoid output gate (doubled q_proj).
    #[serde(default)]
    pub attn_output_gate: bool,

    /// Nested rope_parameters for partial rotation factor etc.
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

/// Nested RoPE configuration (Qwen3.5 style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta_nested")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

/// Wrapper for Qwen3.5's nested config.json format.
#[derive(Deserialize)]
struct Qwen35ConfigWrapper {
    text_config: LlmConfig,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_max_seq_len() -> usize {
    2048
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_conv_kernel_dim() -> usize {
    4
}
fn default_partial_rotary_factor() -> f32 {
    1.0
}
fn default_rope_theta_nested() -> f32 {
    10000.0
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate_size: 5632,
            num_layers: 22,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            model_type: None,
            layer_types: None,
            full_attention_interval: 4,
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 4,
            attn_output_gate: false,
            rope_parameters: None,
        }
    }
}

impl LlmConfig {
    /// Load config from a HuggingFace `config.json` string.
    pub fn from_hf_config(json: &str) -> Result<Self> {
        // Try flat config first (LLaMA-family).
        let flat_result: std::result::Result<LlmConfig, _> = serde_json::from_str(json);

        let mut config = match flat_result {
            Ok(cfg) if cfg.vocab_size > 0 && cfg.hidden_size > 0 => cfg,
            _ => {
                // Try nested Qwen3.5 format: { "text_config": { ... } }
                let wrapper: Qwen35ConfigWrapper =
                    serde_json::from_str(json).map_err(LlmError::Json)?;
                let mut cfg = wrapper.text_config;
                // Inherit top-level tie_word_embeddings if not set in text_config
                if !cfg.tie_word_embeddings && wrapper.tie_word_embeddings {
                    cfg.tie_word_embeddings = wrapper.tie_word_embeddings;
                }
                cfg
            }
        };

        // Compute head_dim if not explicitly set
        if config.head_dim == 0 {
            if config.num_heads == 0 {
                return Err(LlmError::InvalidConfig("num_heads is 0".into()));
            }
            config.head_dim = config.hidden_size / config.num_heads;
        }

        // For Qwen3.5, pick up rope_theta from nested rope_parameters if available
        if let Some(ref rp) = config.rope_parameters {
            if config.rope_theta == default_rope_theta() && rp.rope_theta != default_rope_theta() {
                config.rope_theta = rp.rope_theta;
            }
        }

        config.validate()?;
        Ok(config)
    }

    /// Load config from a HuggingFace `config.json` file.
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_hf_config(&json)
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(LlmError::InvalidConfig("hidden_size is 0".into()));
        }
        if self.num_heads == 0 {
            return Err(LlmError::InvalidConfig("num_heads is 0".into()));
        }
        if self.num_kv_heads == 0 {
            return Err(LlmError::InvalidConfig("num_kv_heads is 0".into()));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(LlmError::InvalidConfig(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        // For Qwen3.5, head_dim * num_heads != hidden_size (e.g. 256*8 = 2048 != 1024).
        // Only validate this for non-Qwen3.5 models.
        if !self.is_qwen35() && self.head_dim * self.num_heads != self.hidden_size {
            return Err(LlmError::InvalidConfig(format!(
                "head_dim ({}) * num_heads ({}) != hidden_size ({})",
                self.head_dim, self.num_heads, self.hidden_size
            )));
        }
        if let Some(ref lt) = self.layer_types {
            if lt.len() != self.num_layers {
                return Err(LlmError::InvalidConfig(format!(
                    "layer_types length ({}) != num_layers ({})",
                    lt.len(),
                    self.num_layers
                )));
            }
        }
        Ok(())
    }

    /// Number of query heads sharing each KV head.
    pub fn gqa_group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Whether this is a Qwen3.5-family model.
    pub fn is_qwen35(&self) -> bool {
        self.model_type
            .as_deref()
            .is_some_and(|t| t.starts_with("qwen3_5"))
    }

    /// Get the layer type for a given layer index.
    pub fn layer_type(&self, idx: usize) -> LayerType {
        if let Some(ref lt) = self.layer_types {
            if idx < lt.len() {
                return match lt[idx].as_str() {
                    "linear_attention" => LayerType::LinearAttention,
                    _ => LayerType::FullAttention,
                };
            }
        }
        LayerType::FullAttention
    }

    /// The fraction of head_dim to which RoPE is applied (1.0 = full).
    pub fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .map_or(1.0, |rp| rp.partial_rotary_factor)
    }

    /// The rotary embedding dimension (head_dim * partial_rotary_factor, rounded to even).
    pub fn rotary_dim(&self) -> usize {
        let raw = (self.head_dim as f32 * self.partial_rotary_factor()) as usize;
        // Must be even for RoPE pairs
        raw & !1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlmConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.gqa_group_size(), 8);
        config.validate().unwrap();
    }

    #[test]
    fn test_from_hf_json() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false
        }"#;
        let config = LlmConfig::from_hf_config(json).unwrap();
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_layers, 22);
        assert!(!config.is_qwen35());
    }

    #[test]
    fn test_invalid_gqa() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 5,
            "intermediate_size": 5632,
            "num_hidden_layers": 22
        }"#;
        assert!(LlmConfig::from_hf_config(json).is_err());
    }

    #[test]
    fn test_roundtrip_serde() {
        let config = LlmConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_qwen35_nested_config() {
        let json = r#"{
            "model_type": "qwen3_5",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 248320,
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "intermediate_size": 3584,
                "num_hidden_layers": 24,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "attn_output_gate": true,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ],
                "rope_parameters": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000
                }
            }
        }"#;
        let config = LlmConfig::from_hf_config(json).unwrap();
        assert!(config.is_qwen35());
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.num_layers, 24);
        assert!(config.tie_word_embeddings);
        assert!(config.attn_output_gate);
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.layer_type(0), LayerType::LinearAttention);
        assert_eq!(config.layer_type(3), LayerType::FullAttention);
        assert_eq!(config.layer_type(23), LayerType::FullAttention);
        assert_eq!(config.partial_rotary_factor(), 0.25);
        assert_eq!(config.rotary_dim(), 64); // 256 * 0.25 = 64
        assert_eq!(config.rope_theta, 10000000.0);
    }

    #[test]
    fn test_layer_type_default() {
        let config = LlmConfig::default();
        assert_eq!(config.layer_type(0), LayerType::FullAttention);
        assert_eq!(config.layer_type(100), LayerType::FullAttention);
    }
}
