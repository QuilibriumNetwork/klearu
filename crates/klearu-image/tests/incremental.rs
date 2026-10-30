use klearu_image::{
    model::{ImageTransformer, ImageTransformerConfig},
    sample::{sample_image_tokens, SampleConfig},
};

fn model() -> ImageTransformer {
    let mut c = ImageTransformerConfig::pilot();
    c.vocab_text = 32;
    c.vocab_image = 16;
    c.bos_token = 32;
    c.sep_image_token = 33;
    c.eos_token = 34;
    c.hidden_size = 32;
    c.num_layers = 2;
    c.num_heads = 4;
    c.mlp_intermediate = 64;
    c.max_text_len = 8;
    c.image_grid_h = 2;
    c.image_grid_w = 3;
    let mut m = ImageTransformer::from_config(c);
    m.init_random(7);
    m
}
#[test]
fn cached_chunks_match_full_forward_and_reject_without_mutating() {
    let m = model();
    let seq = vec![32, 3, 5, 33, 35, 38, 37, 39, 40];
    for chunk_size in [1, 2, 4, 9] {
        let mut cache = m.decode_session();
        let mut seen = 0;
        for chunk in seq.chunks(chunk_size) {
            let actual = cache.forward(chunk).unwrap();
            seen += chunk.len();
            let full = m.forward(&seq[..seen]).unwrap();
            for (a, b) in actual.iter().zip(full[full.len() - 16..].iter()) {
                assert!((a - b).abs() < 2e-5, "{a} vs {b}, chunk={chunk_size}");
            }
            assert!(cache.forward(&[u32::MAX]).is_err());
            assert!(cache.forward(&[]).is_err());
            assert_eq!(cache.len(), seen);
        }
        assert!(cache.forward(&vec![0; m.config.max_seq_len()]).is_err());
        assert_eq!(cache.len(), seq.len());
    }
}
#[test]
fn cached_greedy_sample_matches_uncached_reference() {
    let m = model();
    let mut seq = vec![32, 3, 5, 33];
    let mut expected = Vec::new();
    for _ in 0..6 {
        let logits = m.forward(&seq).unwrap();
        let row = &logits[logits.len() - 16..];
        let id = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        expected.push(id);
        seq.push(m.image_token_to_id(id));
    }
    let actual = sample_image_tokens(
        &m,
        &[3, 5],
        &SampleConfig {
            top_k: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(actual, expected);
}
#[test]
fn adamw_matches_independent_f64_reference() {
    use klearu_image::{
        grad::Gradients,
        optim::{AdamW, AdamWConfig},
    };
    let mut m = model();
    m.embed[0] = 0.75;
    // Large epsilon exercises the bias-correction placement as well as decay.
    let cfg = AdamWConfig {
        lr: 0.03,
        beta1: 0.9,
        beta2: 0.95,
        eps: 0.01,
        weight_decay: 0.2,
    };
    let mut opt = AdamW::new(&m, cfg.clone());
    let mut grad = Gradients::zeros_for(&m);
    let (mut p, mut first, mut second) = (0.75_f64, 0.0_f64, 0.0_f64);
    for (i, g) in [0.0_f32, 0.001, -0.2, 0.5, 0.0].into_iter().enumerate() {
        grad.embed[0] = g;
        opt.step(&mut m, &grad);
        let b1 = cfg.beta1 as f64;
        let b2 = cfg.beta2 as f64;
        first = b1 * first + (1.0 - b1) * g as f64;
        second = b2 * second + (1.0 - b2) * (g as f64).powi(2);
        let mh = first / (1.0 - b1.powi(i as i32 + 1));
        let vh = second / (1.0 - b2.powi(i as i32 + 1));
        p = p * (1.0 - cfg.lr as f64 * cfg.weight_decay as f64)
            - cfg.lr as f64 * mh / (vh.sqrt() + cfg.eps as f64);
        assert!((m.embed[0] as f64 - p).abs() < 1e-6);
    }
}

fn tokenizer(swapped: bool) -> tokenizers::Tokenizer {
    let mut vocab = serde_json::json!({"[UNK]":0,"cat":1,"dog":2});
    if swapped {
        vocab["cat"] = 2.into();
        vocab["dog"] = 1.into();
    }
    let json = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
        "post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}});
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}
#[test]
fn conditioning_identity_survives_checkpoint_and_rejects_mismatch() {
    use klearu_image::{
        checkpoint::{load_model, save_model},
        conditioning::{self, Conditioning},
    };
    let mut m = model();
    let tok = tokenizer(false);
    m.conditioning = conditioning::identity(&tok).unwrap();
    let path = std::env::temp_dir().join(format!(
        "klearu-conditioning-{}.safetensors",
        std::process::id()
    ));
    save_model(&m, &path).unwrap();
    let loaded = load_model(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(loaded.conditioning, m.conditioning);
    let resolved = conditioning::resolve(&loaded.conditioning, None, false)
        .unwrap()
        .unwrap();
    assert_eq!(
        conditioning::encode(&resolved, &m.config, "cat dog").unwrap(),
        vec![1, 2]
    );
    assert!(conditioning::resolve(&loaded.conditioning, Some(tokenizer(true)), false).is_err());
    assert!(conditioning::resolve(&loaded.conditioning, None, true).is_err());
    assert!(conditioning::resolve(&Conditioning::Legacy, None, false).is_err());
    assert!(conditioning::resolve(&Conditioning::Unconditional, None, false).is_err());
    let mut cfg = m.config.clone();
    cfg.vocab_text = 2;
    assert!(conditioning::encode(&tok, &cfg, "cat").is_err()); // whole vocabulary is checked
}
#[test]
fn accumulation_equals_average_of_independent_gradients() {
    use klearu_image::{
        backward::{backward, forward_train},
        distributed::flatten_grad,
        grad::Gradients,
        train::{assemble_batch, TrainExample},
    };
    let m = model();
    let mut combined = Gradients::zeros_for(&m);
    let mut separate = Vec::new();
    for text in [vec![1], vec![2, 3, 4]] {
        let b = assemble_batch(
            &m.config,
            &TrainExample {
                text_tokens: text,
                image_tokens: vec![1, 4, 6, 3, 2, 8],
            },
        )
        .unwrap();
        let c = forward_train(&m, &b.token_ids).unwrap();
        let mut g = Gradients::zeros_for(&m);
        backward(&m, &c, &b.predict_at, &b.targets, &mut g).unwrap();
        separate.push(flatten_grad(&g));
        backward(&m, &c, &b.predict_at, &b.targets, &mut combined).unwrap();
    }
    combined.average_and_clip(2, 1e6).unwrap();
    for ((actual, a), b) in flatten_grad(&combined)
        .iter()
        .zip(&separate[0])
        .zip(&separate[1])
    {
        assert!((actual - (a + b) * 0.5).abs() < 1e-6);
    }
    combined.average_and_clip(1, 0.01).unwrap();
    let norm = flatten_grad(&combined)
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 0.01).abs() < 1e-6);
}

#[test]
fn image_prefix_completion_matches_uncached_reference() {
    use klearu_image::sample::{build_prefix, sample_image_tokens_from_prefix};
    let m = model();
    let cfg = SampleConfig { top_k: 1, ..Default::default() };
    let prefix = vec![3, 7, 2];
    let mut seq = build_prefix(&m.config, &[3, 5]).unwrap();
    seq.extend(prefix.iter().map(|&t| m.image_token_to_id(t)));
    let mut expected = prefix.clone();
    while expected.len() < m.config.image_grid_h * m.config.image_grid_w {
        let logits = m.forward(&seq).unwrap();
        let row = &logits[logits.len() - m.config.vocab_image..];
        let id = row.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        expected.push(id);
        seq.push(m.image_token_to_id(id));
    }
    assert_eq!(sample_image_tokens_from_prefix(&m, &[3,5], &prefix, &cfg).unwrap(), expected);
    assert_eq!(sample_image_tokens_from_prefix(&m, &[3,5], &[], &cfg).unwrap(),
               sample_image_tokens(&m, &[3,5], &cfg).unwrap());
    assert_eq!(sample_image_tokens_from_prefix(&m, &[3,5], &expected, &cfg).unwrap(), expected);
    assert!(sample_image_tokens_from_prefix(&m, &[3,5], &[u32::MAX], &cfg).is_err());
    assert!(sample_image_tokens_from_prefix(&m, &[3,5], &vec![0; expected.len()+1], &cfg).is_err());
}

#[test]
fn stable_image_positions_are_caption_independent_and_cache_safe() {
    let mut m=model(); m.config.stable_image_positions=true;
    let short=vec![32,3,33,35,36,37];
    let long=vec![32,3,5,7,33,35,36,37];
    let (sp,_)=m.config.position_indices(&short,0,None).unwrap();
    let (lp,_)=m.config.position_indices(&long,0,None).unwrap();
    assert_eq!(&sp[2..],&lp[4..]);
    assert_eq!(&sp[2..],&[9,10,11,12]);
    for seq in [&short,&long] {
        for chunk_size in [1,2,3,8] {
            let mut session=m.decode_session(); let mut seen=0;
            for chunk in seq.chunks(chunk_size) {
                let cached=session.forward(chunk).unwrap(); seen+=chunk.len();
                let full=m.forward(&seq[..seen]).unwrap();
                for (a,b) in cached.iter().zip(&full[full.len()-16..]) { assert!((a-b).abs()<2e-5); }
                assert!(session.forward(&[u32::MAX]).is_err());
            }
        }
    }
}

#[test]
fn stable_position_training_gradient_matches_finite_difference() {
    use klearu_image::{backward::{forward_train,backward,forward_train_checkpointed},grad::Gradients,
        train::{TrainExample,assemble_batch,cross_entropy_loss}};
    let mut m=model(); m.config.stable_image_positions=true;
    let batch=assemble_batch(&m.config,&TrainExample{text_tokens:vec![3],image_tokens:vec![1,2,3,4,5,6]}).unwrap();
    let cache=forward_train(&m,&batch.token_ids).unwrap();
    let full=m.forward(&batch.token_ids).unwrap();
    for (a,b) in cache.logits.iter().zip(full) { assert!((a-b).abs()<2e-5); }
    let _=forward_train_checkpointed(&m,&batch.token_ids).unwrap();
    let mut grad=Gradients::zeros_for(&m);
    backward(&m,&cache,&batch.predict_at,&batch.targets,&mut grad).unwrap();
    let d=m.config.hidden_size;
    assert!(grad.pos_embed[3*d..9*d].iter().all(|&g|g==0.0));
    for index in [9*d,9*d+7,10*d+3] {
        let original=m.pos_embed[index]; let eps=1e-3;
        m.pos_embed[index]=original+eps;
        let plus=cross_entropy_loss(&m,&m.forward(&batch.token_ids).unwrap(),&batch).unwrap().0;
        m.pos_embed[index]=original-eps;
        let minus=cross_entropy_loss(&m,&m.forward(&batch.token_ids).unwrap(),&batch).unwrap().0;
        m.pos_embed[index]=original;
        let numeric=(plus-minus)/(2.0*eps);
        assert!((numeric-grad.pos_embed[index]).abs()<5e-4,"{numeric} vs {}",grad.pos_embed[index]);
    }
}

#[test]
fn stable_position_flag_survives_checkpoint() {
    use klearu_image::checkpoint::{save_model,load_model};
    let mut m=model(); m.config.stable_image_positions=true;
    let path=std::env::temp_dir().join(format!("klearu-stable-{}.safetensors",std::process::id()));
    save_model(&m,&path).unwrap();
    let loaded=load_model(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(loaded.config.stable_image_positions);
    assert_eq!(m.forward(&[32,3,33,35]).unwrap(),loaded.forward(&[32,3,33,35]).unwrap());
}
