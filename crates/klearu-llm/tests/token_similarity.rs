#![cfg(feature = "native-io")]

use klearu_llm::similarity::{cosine_similarity, TokenEmbeddings};
use safetensors::{tensor::TensorView, Dtype};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "klearu-similarity-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn weights(&self, file: &str, name: &str, dtype: Dtype) {
        let values = [1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0];
        let bytes: Vec<u8> = values
            .into_iter()
            .flat_map(|v| match dtype {
                Dtype::F32 => v.to_le_bytes().to_vec(),
                Dtype::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                Dtype::BF16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect();
        let view = TensorView::new(dtype, vec![4, 2], &bytes).unwrap();
        safetensors::tensor::serialize_to_file([(name, view)], &None, &self.0.join(file)).unwrap();
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_token-similarity"))
            .arg(&self.0)
            .args(args)
            .output()
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn embedding_formats_pooling_and_validation() {
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        let fixture = Fixture::new();
        fixture.weights("model.safetensors", "model.embed_tokens.weight", dtype);
        let embeddings = TokenEmbeddings::open(&fixture.0, None).unwrap();
        assert_eq!(embeddings.vocab_size(), 4);
        assert_eq!(embeddings.hidden_size(), 2);
        assert_eq!(embeddings.mean_embedding(&[0, 1]).unwrap(), [0.5, 0.5]);
        assert_eq!(
            embeddings.mean_embedding(&[0, 0, 1]).unwrap(),
            [2.0 / 3.0, 1.0 / 3.0]
        );
        assert!(embeddings.mean_embedding(&[]).is_err());
        assert!(embeddings.mean_embedding(&[4]).is_err());
        assert!(
            cosine_similarity(&embeddings.mean_embedding(&[0, 2]).unwrap(), &[1.0, 0.0]).is_err()
        );
        assert!(TokenEmbeddings::open(&fixture.0, Some("missing")).is_err());
    }
}

#[test]
fn cli_ids_text_json_and_errors() {
    let fixture = Fixture::new();
    fixture.weights("model.safetensors", "model.embed_tokens.weight", Dtype::F32);
    let output = fixture.run(&["--a-ids", "0,1", "--b-ids", "0", "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["a_ids"], serde_json::json!([0, 1]));
    assert!((json["cosine_similarity"].as_f64().unwrap() - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);

    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [
                (String::from("cat"), 0),
                (String::from("dog"), 1),
                (String::from("[UNK]"), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer
        .save(fixture.0.join("tokenizer.json"), false)
        .unwrap();
    let output = fixture.run(&["--a", "cat dog", "--b", "cat", "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text_json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json, text_json);

    for args in [
        vec!["--a-ids", "4", "--b-ids", "0"],
        vec!["--a-ids", "3", "--b-ids", "0"],
        vec!["--a", "", "--b", "cat"],
        vec!["--a-ids", "0"],
        vec!["--a-ids", "oops", "--b-ids", "0"],
        vec!["--a", "cat", "--a-ids", "0", "--b-ids", "1"],
    ] {
        let output = fixture.run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    }
}

#[test]
fn shard_discovery_custom_names_and_malformed_files() {
    let fixture = Fixture::new();
    fixture.weights("model-00001.safetensors", "unrelated.weight", Dtype::F32);
    fixture.weights(
        "model-00002.safetensors",
        "model.language_model.embed_tokens.weight",
        Dtype::BF16,
    );
    let embeddings = TokenEmbeddings::open(&fixture.0, None).unwrap();
    assert_eq!(embeddings.mean_embedding(&[0]).unwrap(), [1.0, 0.0]);
    let custom = TokenEmbeddings::open(
        &fixture.0.join("model-00001.safetensors"),
        Some("unrelated.weight"),
    )
    .unwrap();
    assert_eq!(custom.mean_embedding(&[1]).unwrap(), [0.0, 1.0]);
    fixture.weights(
        "duplicate.safetensors",
        "model.embed_tokens.weight",
        Dtype::F32,
    );
    assert!(TokenEmbeddings::open(&fixture.0, None).is_err());
    std::fs::write(fixture.0.join("bad.safetensors"), b"invalid").unwrap();
    assert!(TokenEmbeddings::open(&fixture.0.join("bad.safetensors"), None).is_err());
}
