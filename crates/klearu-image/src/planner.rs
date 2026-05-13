//! Reasoning planner prefix — the "thinking before drawing" pre-pass.
//!
//! ChatGPT Images 2.0's quality jump over earlier models comes largely
//! from a reasoning step that runs BEFORE pixel generation: parse the
//! prompt, identify objects/relations/constraints, output a structured
//! scratchpad, then condition the image transformer on it. We mirror
//! that with a small `klearu-llm` model producing a plan, prepended to
//! the BPE text tokens.
//!
//! ## Architecture
//!
//! User prompt
//!   │
//!   ▼
//! ┌──────────────────┐
//! │ klearu-llm 125M  │ (any small causal LM; reused as-is)
//! └──────────────────┘
//!   │ generates: "<plan> subjects: [cat]; layout: centered; style: ... </plan>"
//!   ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │ BPE tokenize: <plan> tokens... </plan> user prompt tokens │
//! └─────────────────────────────────────────────────────────┘
//!   │
//!   ▼  (fits in `max_text_len`; the image transformer doesn't see
//!       the distinction — it's just more text prefix)
//! ┌──────────────────┐
//! │ ImageTransformer │
//! └──────────────────┘
//!
//! ## Why this works at small scale
//!
//! The planner doesn't need to be huge. At our scale (50M-param image
//! model, 256-token grid), even a ~30M-param scratchpad LLM with a
//! handful of layers can:
//!   - Disambiguate compound prompts ("a red cat AND a blue dog").
//!   - Enforce object counts.
//!   - Pre-decompose stylistic modifiers from object descriptions.
//!
//! These are exactly the failure modes that show up at our scale, where
//! the image transformer alone hallucinates objects or merges modifiers.
//!
//! ## What's in this module
//!
//! We don't implement the planner model itself — it's just any
//! `klearu_llm::Model` instance. This module ships:
//!   - `Plan` struct: structured representation a hand-coded or
//!     LLM-generated plan can land in.
//!   - `plan_to_tokens`: serialise a `Plan` into BPE-tokenizable text.
//!   - `assemble_planned_prefix`: build the final text-token prefix
//!     consumed by the image transformer's prompt position.
//!
//! Plug-in semantics: the user runs their planner LLM independently
//! (klearu-llm crate has all the infrastructure), passes the resulting
//! text to BPE, then feeds the resulting `Vec<u32>` through this
//! module's `assemble_planned_prefix` to slot it correctly under the
//! image model's `max_text_len`.

use crate::error::{ImageGenError, Result};
use crate::model::ImageTransformerConfig;

/// A structured pre-image plan. Optional fields — caller fills what it
/// has. Sensible defaults exist for everything.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// Primary subjects in the image. Order matters; first is the focal.
    pub subjects: Vec<String>,
    /// Spatial layout hints ("centered", "left side", "top half", etc.).
    pub layout: Option<String>,
    /// Style modifier ("photorealistic", "anime", "watercolor", …).
    pub style: Option<String>,
    /// Lighting hint ("golden hour", "studio", "moonlit", …).
    pub lighting: Option<String>,
    /// Free-form notes. Used as a catch-all if the LLM produces extra
    /// context that doesn't fit the structured fields.
    pub notes: Option<String>,
}

impl Plan {
    /// Render the plan to a string suitable for BPE tokenization. The
    /// format is `<plan>key: value; key: value; …</plan>` — compact,
    /// stable, and parses back into structured form via simple regex.
    pub fn to_text(&self) -> String {
        let mut buf = String::from("<plan>");
        if !self.subjects.is_empty() {
            buf.push_str(" subjects: [");
            for (i, s) in self.subjects.iter().enumerate() {
                if i > 0 { buf.push_str(", "); }
                buf.push_str(s);
            }
            buf.push_str("];");
        }
        if let Some(s) = &self.layout {
            buf.push_str(&format!(" layout: {s};"));
        }
        if let Some(s) = &self.style {
            buf.push_str(&format!(" style: {s};"));
        }
        if let Some(s) = &self.lighting {
            buf.push_str(&format!(" lighting: {s};"));
        }
        if let Some(s) = &self.notes {
            buf.push_str(&format!(" notes: {s};"));
        }
        buf.push_str(" </plan>");
        buf
    }

    /// Approximate token count for a typical BPE tokenizer. Useful for
    /// deciding whether the plan fits inside `max_text_len` with room
    /// to spare for the user's prompt. Heuristic — true count comes from
    /// the actual tokenizer.
    pub fn estimated_token_count(&self) -> usize {
        let text_len = self.to_text().len();
        // Rough rule of thumb: 4 chars per BPE token, plus ~10 token
        // overhead for the <plan>…</plan> wrapper.
        text_len / 4 + 10
    }
}

/// Assemble the final text-token prefix for the image transformer:
/// `[ plan_tokens... , user_text_tokens... ]`. Truncates the user's
/// portion if the combined length would exceed `max_text_len`. The
/// plan is kept intact at the front — it's higher-value signal than
/// the trailing words of a long user prompt.
pub fn assemble_planned_prefix(
    cfg: &ImageTransformerConfig,
    plan_tokens: &[u32],
    user_text_tokens: &[u32],
) -> Result<Vec<u32>> {
    // Validate id ranges (only text-vocab ids allowed in either slice).
    for (i, &t) in plan_tokens.iter().enumerate() {
        if (t as usize) >= cfg.vocab_text {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("plan_tokens id < vocab_text = {}", cfg.vocab_text),
                got: format!("id={t} at plan position {i}"),
            });
        }
    }
    for (i, &t) in user_text_tokens.iter().enumerate() {
        if (t as usize) >= cfg.vocab_text {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("user_text_tokens id < vocab_text = {}", cfg.vocab_text),
                got: format!("id={t} at user position {i}"),
            });
        }
    }

    let max = cfg.max_text_len;
    if plan_tokens.len() >= max {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("plan_tokens ≤ max_text_len = {max} (leaving room for user prompt)"),
            got: format!("{}", plan_tokens.len()),
        });
    }

    // Keep the plan in full; truncate the user portion to fit.
    let remaining = max - plan_tokens.len();
    let user_kept = &user_text_tokens[..remaining.min(user_text_tokens.len())];
    let mut out = Vec::with_capacity(plan_tokens.len() + user_kept.len());
    out.extend_from_slice(plan_tokens);
    out.extend_from_slice(user_kept);
    Ok(out)
}

// ============================================================================
// LLM-driven planner integration
// ============================================================================
//
// Wraps a klearu-llm `Pipeline` to turn a user prompt into a structured plan
// at inference time. The system-prompt template asks the LLM to emit ONLY
// the canonical `<plan>…</plan>` form so the parser can recover a `Plan`
// reliably without depending on JSON adherence.
//
// Usage is opt-in: the planner module's core types (`Plan`,
// `assemble_planned_prefix`) work standalone without an LLM. If you have a
// hand-written plan, skip `ReasoningPlanner` entirely.

use klearu_llm::generate::pipeline::{GenerateConfig, Pipeline};
use klearu_llm::generate::sampler::SamplerConfig;

/// Wraps a klearu-llm pipeline so a user prompt becomes a `Plan` via
/// model generation, then BPE tokens via the image transformer's
/// tokenizer.
pub struct ReasoningPlanner {
    pub pipeline: Pipeline,
    /// System prompt prefix used for every plan generation. Templated
    /// with the user prompt at `{prompt}`.
    pub system_template: String,
    /// Generation knobs (low temperature recommended — we want structured
    /// output, not creative variation).
    pub gen_config: GenerateConfig,
}

impl ReasoningPlanner {
    /// Default system template: instructs the LLM to emit `<plan>…</plan>`
    /// in the exact format `Plan::to_text` produces. Override via
    /// `system_template` for domain-specific planners.
    pub const DEFAULT_TEMPLATE: &'static str = "\
You are an image-generation planner. For the user prompt below, produce \
a structured plan listing the primary subjects (most important first), \
the spatial layout, style, and lighting. Reply with EXACTLY this format \
and nothing else:\n\
<plan> subjects: [item1, item2]; layout: ...; style: ...; lighting: ... </plan>\n\
\n\
User prompt: {prompt}\n\
Plan: ";

    pub fn new(pipeline: Pipeline) -> Self {
        Self {
            pipeline,
            system_template: Self::DEFAULT_TEMPLATE.into(),
            gen_config: GenerateConfig {
                max_new_tokens: 128,
                sampler: SamplerConfig {
                    temperature: 0.2,
                    top_k: 16,
                    top_p: 0.95,
                    repetition_penalty: 1.05,
                },
                ..GenerateConfig::default()
            },
        }
    }

    /// Run the LLM to produce a Plan for `user_prompt`. The model's
    /// raw output is parsed via `parse_plan_text`; if parsing fails the
    /// returned Plan is empty (caller can decide whether to fall back to
    /// the user prompt directly).
    pub fn plan(&mut self, user_prompt: &str, rng_seed: u64) -> Result<Plan> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
        let filled = self.system_template.replace("{prompt}", user_prompt);
        let raw = self.pipeline.generate(&filled, &self.gen_config, &mut rng)
            .map_err(|e| ImageGenError::Unsupported(format!("planner: {e}")))?;
        Ok(parse_plan_text(&raw))
    }
}

/// Parse a `<plan>…</plan>` blob back into a `Plan`. Tolerates extra text
/// before / after the tag. Missing fields stay None. Designed to be the
/// inverse of `Plan::to_text` modulo whitespace.
pub fn parse_plan_text(raw: &str) -> Plan {
    let mut plan = Plan::default();
    // Find the <plan>…</plan> span.
    let body = match (raw.find("<plan>"), raw.find("</plan>")) {
        (Some(a), Some(b)) if b > a => &raw[a + "<plan>".len()..b],
        _ => return plan, // No tags found — return empty plan.
    };
    // Each field is `key: value;` separated. Subjects are bracketed.
    for chunk in body.split(';') {
        let chunk = chunk.trim();
        if chunk.is_empty() { continue; }
        let (key, value) = match chunk.split_once(':') {
            Some((k, v)) => (k.trim().to_ascii_lowercase(), v.trim()),
            None => continue,
        };
        match key.as_str() {
            "subjects" => {
                // Strip surrounding brackets if present.
                let v = value.trim_start_matches('[').trim_end_matches(']');
                plan.subjects = v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "layout"   => plan.layout = Some(value.into()),
            "style"    => plan.style = Some(value.into()),
            "lighting" => plan.lighting = Some(value.into()),
            "notes"    => plan.notes = Some(value.into()),
            _ => {}
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip() {
        let p = Plan {
            subjects: vec!["a cat".into(), "a hat".into()],
            layout: Some("centered".into()),
            style: Some("watercolor".into()),
            lighting: Some("soft".into()),
            notes: None,
        };
        let parsed = parse_plan_text(&p.to_text());
        assert_eq!(parsed.subjects, p.subjects);
        assert_eq!(parsed.layout, p.layout);
        assert_eq!(parsed.style, p.style);
        assert_eq!(parsed.lighting, p.lighting);
        assert_eq!(parsed.notes, None);
    }

    #[test]
    fn parse_tolerates_surrounding_text() {
        let raw = "Sure! Here is the plan: <plan> subjects: [dog]; \
            style: oil paint </plan> Hope that helps!";
        let p = parse_plan_text(raw);
        assert_eq!(p.subjects, vec!["dog".to_string()]);
        assert_eq!(p.style.as_deref(), Some("oil paint"));
        assert!(p.layout.is_none());
    }

    #[test]
    fn parse_missing_tags_returns_empty_plan() {
        let p = parse_plan_text("no plan here");
        assert!(p.subjects.is_empty());
        assert!(p.layout.is_none());
    }

    #[test]
    fn plan_to_text_roundtrip_structure() {
        let p = Plan {
            subjects: vec!["a black cat".into(), "a red ball".into()],
            layout: Some("centered".into()),
            style: Some("photorealistic".into()),
            lighting: Some("soft daylight".into()),
            notes: None,
        };
        let s = p.to_text();
        assert!(s.starts_with("<plan>") && s.ends_with("</plan>"));
        assert!(s.contains("subjects: [a black cat, a red ball]"));
        assert!(s.contains("layout: centered"));
        assert!(s.contains("style: photorealistic"));
        assert!(!s.contains("notes:"));
    }

    #[test]
    fn empty_plan_renders_compactly() {
        let p = Plan::default();
        let s = p.to_text();
        assert_eq!(s, "<plan> </plan>");
    }

    #[test]
    fn assembled_prefix_truncates_user_if_needed() {
        let cfg = ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 16, num_layers: 1, num_heads: 4,
            mlp_intermediate: 32,
            max_text_len: 10,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let plan = vec![1_u32, 2, 3, 4];                  // 4 tokens
        let user = vec![10_u32, 11, 12, 13, 14, 15, 16, 17]; // 8 tokens
        let merged = assemble_planned_prefix(&cfg, &plan, &user).expect("assemble");
        // Total budget = 10, plan = 4, so user gets first 6 of its 8.
        assert_eq!(merged.len(), 10);
        assert_eq!(&merged[..4], plan.as_slice());
        assert_eq!(&merged[4..], &user[..6]);
    }

    #[test]
    fn plan_too_long_errors() {
        let cfg = ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 16, num_layers: 1, num_heads: 4,
            mlp_intermediate: 32,
            max_text_len: 5,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let plan = vec![1_u32; 10]; // longer than max_text_len
        let user = vec![5_u32];
        assert!(assemble_planned_prefix(&cfg, &plan, &user).is_err());
    }

    #[test]
    fn rejects_image_vocab_ids_in_text_slots() {
        let cfg = ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 16, num_layers: 1, num_heads: 4,
            mlp_intermediate: 32,
            max_text_len: 10,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let bad_plan = vec![cfg.vocab_text as u32 + 5];
        let ok_user = vec![5_u32];
        assert!(assemble_planned_prefix(&cfg, &bad_plan, &ok_user).is_err());
    }
}
