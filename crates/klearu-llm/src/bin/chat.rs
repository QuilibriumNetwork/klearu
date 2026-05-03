use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};

use rand::SeedableRng;

use klearu_llm::generate::chat_template::{ChatMessage, ChatTemplate};
use klearu_llm::generate::pipeline::{GenerateConfig, Pipeline, detect_eos_tokens};
use klearu_llm::generate::sampler::SamplerConfig;
#[cfg(feature = "sparse")]
use klearu_llm::generate::sparse_pipeline::SparsePipeline;

// --- Per-user steering-vector support (frozen-body continual learning) ---
//
// Maintains a single hidden_size-float "user vector" + Welford-style
// running per-class means alongside it. /+ /- commands during chat update
// the running stats and recompute v = μ⁺ − μ⁻; v is added scaled by α
// to every generated token's post-final-norm hidden via Pipeline's
// `generate_streaming_steered`.

const VEC_MAGIC: &[u8] = b"USERVEC1";
const STATE_MAGIC: &[u8] = b"USRSTAT1";

struct UserVector {
    hidden_size: usize,
    v: Vec<f32>,
}
impl UserVector {
    fn empty(hidden: usize) -> Self { Self { hidden_size: hidden, v: vec![0.0; hidden] } }
    fn load(path: &Path, expected: usize) -> std::io::Result<Self> {
        let mut r = std::fs::File::open(path)?;
        let mut magic = [0u8; 8]; r.read_exact(&mut magic)?;
        if magic != VEC_MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad vec magic"));
        }
        let mut buf4 = [0u8; 4]; r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        if n != expected {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "hidden size mismatch"));
        }
        let mut v = vec![0.0f32; n];
        for x in v.iter_mut() { r.read_exact(&mut buf4)?; *x = f32::from_le_bytes(buf4); }
        Ok(Self { hidden_size: n, v })
    }
    fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(VEC_MAGIC)?;
        w.write_all(&(self.hidden_size as u32).to_le_bytes())?;
        for x in &self.v { w.write_all(&x.to_le_bytes())?; }
        Ok(())
    }
    fn norm(&self) -> f32 { self.v.iter().map(|x| x * x).sum::<f32>().sqrt() }
}
struct WelfordState {
    n_pos: u32, n_neg: u32, mean_pos: Vec<f32>, mean_neg: Vec<f32>,
}
impl WelfordState {
    fn empty(h: usize) -> Self { Self { n_pos: 0, n_neg: 0, mean_pos: vec![0.0; h], mean_neg: vec![0.0; h] } }
    fn load(path: &Path, expected: usize) -> std::io::Result<Self> {
        let mut r = std::fs::File::open(path)?;
        let mut magic = [0u8; 8]; r.read_exact(&mut magic)?;
        if magic != STATE_MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad state magic"));
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?; let n_pos = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?; let n_neg = u32::from_le_bytes(buf4);
        let mut mean_pos = vec![0.0f32; expected];
        for v in mean_pos.iter_mut() { r.read_exact(&mut buf4)?; *v = f32::from_le_bytes(buf4); }
        let mut mean_neg = vec![0.0f32; expected];
        for v in mean_neg.iter_mut() { r.read_exact(&mut buf4)?; *v = f32::from_le_bytes(buf4); }
        Ok(Self { n_pos, n_neg, mean_pos, mean_neg })
    }
    fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(STATE_MAGIC)?;
        w.write_all(&self.n_pos.to_le_bytes())?;
        w.write_all(&self.n_neg.to_le_bytes())?;
        for v in &self.mean_pos { w.write_all(&v.to_le_bytes())?; }
        for v in &self.mean_neg { w.write_all(&v.to_le_bytes())?; }
        Ok(())
    }
    fn observe(&mut self, label: i8, h: &[f32]) {
        let (n, mean) = if label > 0 { (&mut self.n_pos, &mut self.mean_pos) }
                         else { (&mut self.n_neg, &mut self.mean_neg) };
        *n += 1;
        let inv = 1.0f32 / *n as f32;
        for (i, &v) in h.iter().enumerate() {
            mean[i] += (v - mean[i]) * inv;
        }
    }
    fn vector(&self) -> Vec<f32> {
        if self.n_pos == 0 || self.n_neg == 0 { return vec![0.0; self.mean_pos.len()]; }
        self.mean_pos.iter().zip(self.mean_neg.iter()).map(|(p, n)| p - n).collect()
    }
}

// Power-iteration PCA: compute top-K right singular vectors of the
// data matrix X (rows = samples, cols = hidden_size). Uses the
// X X^T trick: when n_samples ≪ hidden_size, eigendecompose the
// small n×n Gram matrix instead of the d×d covariance. Returns
// (sigma_k, v_k) pairs where v_k is normalised in hidden space.
fn pca_top_k(rows: &[Vec<f32>], k: usize) -> Vec<(f32, Vec<f32>)> {
    let n = rows.len();
    if n == 0 || k == 0 { return Vec::new(); }
    let d = rows[0].len();
    if d == 0 { return Vec::new(); }

    // Build n × n Gram matrix M = X X^T (symmetric, PSD).
    let mut m = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i..n {
            let dot: f32 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| a * b).sum();
            m[i][j] = dot;
            m[j][i] = dot;
        }
    }

    let k = k.min(n);
    let mut result: Vec<(f32, Vec<f32>)> = Vec::with_capacity(k);
    for component_idx in 0..k {
        // Power iteration on m to find its top eigenvector.
        let mut u = vec![1.0f32 / (n as f32).sqrt(); n];
        if component_idx < n {
            // Diversify init by using a row vector for variation.
            u[component_idx] = 1.0;
            let n2: f32 = u.iter().map(|x| x*x).sum::<f32>().sqrt();
            if n2 > 0.0 { for v in u.iter_mut() { *v /= n2; } }
        }
        let mut lambda = 0.0f32;
        for _ in 0..80 {
            let mut next = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    next[i] += m[i][j] * u[j];
                }
            }
            let nrm: f32 = next.iter().map(|x| x*x).sum::<f32>().sqrt();
            if nrm < 1e-9 { lambda = 0.0; break; }
            for v in next.iter_mut() { *v /= nrm; }
            // λ ≈ u^T M u (Rayleigh quotient)
            let mut mu = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n { mu[i] += m[i][j] * next[j]; }
            }
            lambda = next.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f32>();
            u = next;
        }
        if lambda <= 0.0 { break; }
        // Right singular vector: v_k = (X^T u_k) / σ_k
        let mut v = vec![0.0f32; d];
        for j in 0..d {
            for i in 0..n { v[j] += u[i] * rows[i][j]; }
        }
        let sigma = v.iter().map(|x| x*x).sum::<f32>().sqrt();
        if sigma > 1e-9 {
            for x in v.iter_mut() { *x /= sigma; }
        }
        result.push((sigma, v));
        // Deflate: M ← M − λ u u^T
        for i in 0..n {
            for j in 0..n {
                m[i][j] -= lambda * u[i] * u[j];
            }
        }
    }
    result
}

// Read just (text, hidden) pairs from a klearu Capture file (.bin).
// Format: 4-byte magic "KLTI", u32 version (=2), length-prefixed model_id,
// u32 hidden_size, u32 gate_size, u32 target_layer (MAX=None), u32 num_samples,
// then per sample: length-prefixed text, u32 token_count, hidden_size×f32 hidden,
// gate_size×f32 gate_preact.
fn load_probe_dict(path: &Path, expected_hidden: usize) -> std::io::Result<Vec<(String, Vec<f32>)>> {
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
    if &magic != b"KLTI" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a klearu capture"));
    }
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?; let _version = u32::from_le_bytes(buf4);
    // model_id
    r.read_exact(&mut buf4)?; let mid_len = u32::from_le_bytes(buf4) as usize;
    let mut mid = vec![0u8; mid_len]; r.read_exact(&mut mid)?;
    r.read_exact(&mut buf4)?; let hidden_size = u32::from_le_bytes(buf4) as usize;
    if hidden_size != expected_hidden {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("dict hidden_size {hidden_size} ≠ model {expected_hidden}")));
    }
    r.read_exact(&mut buf4)?; let gate_size = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4)?; let _target_layer = u32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?; let num_samples = u32::from_le_bytes(buf4) as usize;

    let mut out = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        r.read_exact(&mut buf4)?;
        let text_len = u32::from_le_bytes(buf4) as usize;
        let mut tbuf = vec![0u8; text_len]; r.read_exact(&mut tbuf)?;
        let text = String::from_utf8(tbuf).unwrap_or_default();
        r.read_exact(&mut buf4)?; // token_count
        let mut h = vec![0.0f32; hidden_size];
        for v in h.iter_mut() { r.read_exact(&mut buf4)?; *v = f32::from_le_bytes(buf4); }
        for _ in 0..gate_size { r.read_exact(&mut buf4)?; }
        out.push((text, h));
    }
    Ok(out)
}

// Load a steering vector from the vector_library on-disk format.
// Format mirrors UserVector: 8-byte magic + 4-byte hidden_size + f32×hidden_size.
// vector_library writes a Translator-format .vec which is more elaborate,
// so both are accepted: Translator-format is tried first, then UserVector-format.
fn load_persona_vec(path: &Path, expected_hidden: usize) -> std::io::Result<Vec<f32>> {
    let mut r = std::fs::File::open(path)?;
    // First peek: 8 bytes (UserVector magic is 8 bytes; Translator uses 4-byte "KLTT" + version u32).
    let mut head = [0u8; 8];
    r.read_exact(&mut head)?;
    if &head == VEC_MAGIC {
        // UserVector format: [magic 8][hidden_size u32][f32 × hidden_size]
        let mut buf4 = [0u8; 4]; r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        if n != expected_hidden {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("persona hidden_size {n} ≠ model {expected_hidden}")));
        }
        let mut v = vec![0.0f32; n];
        for x in v.iter_mut() { r.read_exact(&mut buf4)?; *x = f32::from_le_bytes(buf4); }
        return Ok(v);
    }
    // Translator format (vector_library writes this): "KLTT"[4] + version u32 + ...
    if &head[..4] == b"KLTT" {
        // magic[4] + version[4] = 8 bytes have already been consumed.
        let mut buf4 = [0u8; 4];
        // skip source_id and target_id
        for _ in 0..2 {
            r.read_exact(&mut buf4)?;
            let n = u32::from_le_bytes(buf4) as usize;
            let mut s = vec![0u8; n]; r.read_exact(&mut s)?;
        }
        r.read_exact(&mut buf4)?;
        let source_hidden = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let target_hidden = u32::from_le_bytes(buf4) as usize;
        if target_hidden != expected_hidden {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("persona target_hidden {target_hidden} ≠ model {expected_hidden}")));
        }
        // skip source_layer (u32), target_layer (u32), lambda (f32), n_train (u32)
        for _ in 0..4 { r.read_exact(&mut buf4)?; }
        // skip a_means
        for _ in 0..source_hidden { r.read_exact(&mut buf4)?; }
        // read b_means — this IS the steering vector for vector_library output
        let mut v = vec![0.0f32; target_hidden];
        for x in v.iter_mut() { r.read_exact(&mut buf4)?; *x = f32::from_le_bytes(buf4); }
        return Ok(v);
    }
    Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
        format!("unknown vector format magic: {head:?}")))
}

macro_rules! run_chat_loop {
    ($pipeline:expr, $template:expr, $config:expr, $system_msg:expr, $user_state:expr, $library:expr, $calibration_name:expr, $profiles_path:expr, $cache_path:expr, $cache_threshold:expr) => {{
        let mut rng = rand::rngs::StdRng::from_entropy();
        let mut history: Vec<ChatMessage> = Vec::new();

        if let Some(sys) = &$system_msg {
            history.push(ChatMessage::system(sys.clone()));
            eprintln!("System: {sys}");
        }

        eprintln!("Ready. Type a message, or /+ /- /alpha N /info /quit. Ctrl-D to quit.\n");

        let stdin = io::stdin();
        let hidden_size = $pipeline.model.config.hidden_size;
        let (vec_path, state_path, mut user_vec, mut state, mut alpha) = match &$user_state {
            Some((vp, sp, v, s, a)) => (vp.clone(), sp.clone(), UserVector { hidden_size: v.hidden_size, v: v.v.clone() }, WelfordState { n_pos: s.n_pos, n_neg: s.n_neg, mean_pos: s.mean_pos.clone(), mean_neg: s.mean_neg.clone() }, *a),
            None => (PathBuf::new(), PathBuf::new(), UserVector::empty(hidden_size), WelfordState::empty(hidden_size), 0.0),
        };
        let steering_enabled = $user_state.is_some();
        let mut last_response_hidden: Option<Vec<f32>> = None;

        // Persona library: directory of <name>.vec files. /persona <name>
        // sets `active_persona` to the loaded vector; total steering becomes
        // α * (user_vec + persona_vec).
        let persona_library: Option<PathBuf> = $library.clone();
        let mut active_persona: Option<(String, Vec<f32>)> = None;

        // Profiles: <name> => spec like "register:1.5,brevity:1.0".
        // /context <name> activates the named profile by composing the
        // referenced vectors from the persona library at the given weights.
        // Probe-cache layer: pre-generation cosine-NN over a captured
        // (prompt, response) dictionary. On a high-confidence hit, return
        // the cached response immediately, skipping generation.
        let probe_dict: Vec<(String, Vec<f32>, f32)> = match &$cache_path {
            Some(p) => match load_probe_dict(p, hidden_size) {
                Ok(entries) => {
                    eprintln!("(probe-cache: {} entries from {} at threshold {})",
                        entries.len(), p.display(), $cache_threshold);
                    entries.into_iter()
                        .map(|(t, h)| { let n = h.iter().map(|x| x*x).sum::<f32>().sqrt(); (t, h, n) })
                        .collect()
                }
                Err(e) => { eprintln!("(failed to load --cache: {e})"); Vec::new() }
            },
            None => Vec::new(),
        };
        let cache_threshold_f: f32 = $cache_threshold;

        // /topic: maintain a list of user-message hiddens for PCA-based
        // conversation table-of-contents.
        let mut user_msg_hiddens: Vec<Vec<f32>> = Vec::new();
        let mut user_msg_texts: Vec<String> = Vec::new();

        let profiles: std::collections::HashMap<String, Vec<(String, f32)>> = match &$profiles_path {
            Some(p) => {
                let txt = std::fs::read_to_string(p).unwrap_or_default();
                let mut m = std::collections::HashMap::new();
                for line in txt.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') { continue; }
                    let (name, spec) = match line.split_once("=>") {
                        Some(x) => x,
                        None => continue,
                    };
                    let parts: Vec<(String, f32)> = spec.split(',')
                        .filter_map(|p| {
                            let (k, v) = p.split_once(':')?;
                            Some((k.trim().to_string(), v.trim().parse::<f32>().ok()?))
                        })
                        .collect();
                    m.insert(name.trim().to_string(), parts);
                }
                eprintln!("(loaded {} profiles from {})", m.len(), p.display());
                m
            }
            None => std::collections::HashMap::new(),
        };

        // --calibration NAME: at startup, load <library>/<NAME>.vec as the
        // initial active_persona. Equivalent to running /persona NAME.
        if let (Some(name), Some(lib)) = (&$calibration_name, &persona_library) {
            let p = lib.join(format!("{name}.vec"));
            match load_persona_vec(&p, hidden_size) {
                Ok(v) => {
                    let n2: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                    eprintln!("Calibration {name} loaded: ‖v‖₂={n2:.4}");
                    active_persona = Some((name.clone(), v));
                }
                Err(e) => eprintln!("(failed to load calibration {name} at {}: {e})", p.display()),
            }
        } else if $calibration_name.is_some() && persona_library.is_none() {
            eprintln!("(--calibration requires --library)");
        }

        loop {
            eprint!("> ");
            io::stderr().flush().unwrap();

            let mut input = String::new();
            if stdin.lock().read_line(&mut input).unwrap() == 0 {
                eprintln!();
                break;
            }
            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            // /command handling (only meaningful when --user is set).
            if input.starts_with('/') {
                let mut parts = input.splitn(2, char::is_whitespace);
                let cmd = parts.next().unwrap();
                let arg = parts.next().unwrap_or("");
                match cmd {
                    "/quit" | "/q" | "/exit" => break,
                    "/help" => {
                        eprintln!("commands: /+ /- /alpha <f32> /info /clear /quit");
                        continue;
                    }
                    "/clear" => { history.clear(); if let Some(sys) = &$system_msg { history.push(ChatMessage::system(sys.clone())); } eprintln!("(cleared)"); continue; }
                    "/alpha" => {
                        if let Ok(a) = arg.parse::<f32>() { alpha = a; eprintln!("(α = {alpha})"); }
                        else { eprintln!("(usage: /alpha <f32>)"); }
                        continue;
                    }
                    "/info" => {
                        let persona_str = match &active_persona {
                            Some((n, v)) => format!(", persona={n} ‖={:.4}",
                                v.iter().map(|x| x*x).sum::<f32>().sqrt()),
                            None => String::new(),
                        };
                        eprintln!("(α={alpha}, ‖user‖₂={:.4}, n_pos={}, n_neg={}{persona_str})",
                            user_vec.norm(), state.n_pos, state.n_neg);
                        continue;
                    }
                    "/topic" => {
                        if user_msg_hiddens.len() < 3 {
                            eprintln!("(need ≥3 user messages; have {})", user_msg_hiddens.len());
                            continue;
                        }
                        // Mean-centre.
                        let mut means = vec![0.0f32; hidden_size];
                        for r in &user_msg_hiddens {
                            for (i, &v) in r.iter().enumerate() { means[i] += v; }
                        }
                        let inv = 1.0f32 / user_msg_hiddens.len() as f32;
                        for v in means.iter_mut() { *v *= inv; }
                        let centred: Vec<Vec<f32>> = user_msg_hiddens.iter()
                            .map(|r| r.iter().zip(means.iter()).map(|(a, b)| a - b).collect())
                            .collect();
                        let triplets = pca_top_k(&centred, 3);
                        eprintln!("(topics over {} messages)", user_msg_hiddens.len());
                        for (k, (sigma, v)) in triplets.iter().enumerate() {
                            // Decode top tokens for ±v.
                            let plus = $pipeline.model.apply_lm_head(v);
                            let neg: Vec<f32> = v.iter().map(|x| -x).collect();
                            let neg_logits = $pipeline.model.apply_lm_head(&neg);
                            let mut p_pairs: Vec<(usize, f32)> = plus.iter().copied().enumerate().collect();
                            p_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                            let mut n_pairs: Vec<(usize, f32)> = neg_logits.iter().copied().enumerate().collect();
                            n_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                            let p_top: Vec<String> = p_pairs.iter().take(5)
                                .map(|(i, _)| $pipeline.tokenizer.decode(&[*i as u32]).unwrap_or_default())
                                .collect();
                            let n_top: Vec<String> = n_pairs.iter().take(5)
                                .map(|(i, _)| $pipeline.tokenizer.decode(&[*i as u32]).unwrap_or_default())
                                .collect();
                            eprintln!("  topic {k} σ={sigma:.2}: +{p_top:?} −{n_top:?}");
                            // Per-message loadings: project each centred row onto v_k.
                            let mut loadings: Vec<(usize, f32)> = centred.iter().enumerate()
                                .map(|(i, row)| {
                                    let proj: f32 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                                    (i, proj)
                                }).collect();
                            loadings.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
                            for (idx, proj) in loadings.iter().take(2) {
                                let t: String = user_msg_texts[*idx].chars().take(48).collect();
                                eprintln!("    {proj:>+8.3}  {:?}", t);
                            }
                        }
                        continue;
                    }
                    "/context" => {
                        if profiles.is_empty() {
                            eprintln!("(no --profiles configured)");
                            continue;
                        }
                        if arg.is_empty() || arg == "list" {
                            eprintln!("(profiles)");
                            for (name, spec) in &profiles {
                                let s: Vec<String> = spec.iter()
                                    .map(|(l, w)| format!("{l}×{w:+.2}"))
                                    .collect();
                                eprintln!("  {name}: {}", s.join(" + "));
                            }
                            continue;
                        }
                        if arg == "none" {
                            active_persona = None;
                            eprintln!("(context cleared)");
                            continue;
                        }
                        let spec = match profiles.get(arg) {
                            Some(s) => s.clone(),
                            None => { eprintln!("(no profile named {arg})"); continue; }
                        };
                        let lib = match &persona_library {
                            Some(l) => l.clone(),
                            None => { eprintln!("(no --library configured)"); continue; }
                        };
                        // Compose: load each referenced vector and weight-sum.
                        let mut combined = vec![0.0f32; hidden_size];
                        let mut applied: Vec<String> = Vec::new();
                        for (label, weight) in &spec {
                            let p = lib.join(format!("{label}.vec"));
                            match load_persona_vec(&p, hidden_size) {
                                Ok(v) => {
                                    for j in 0..hidden_size {
                                        combined[j] += weight * v[j];
                                    }
                                    applied.push(format!("{label}×{weight:+.2}"));
                                }
                                Err(e) => eprintln!("(skip {label}: {e})"),
                            }
                        }
                        let n2: f32 = combined.iter().map(|x| x*x).sum::<f32>().sqrt();
                        eprintln!("(context {arg} active: [{}], ‖={:.4})", applied.join(" + "), n2);
                        active_persona = Some((arg.to_string(), combined));
                        continue;
                    }
                    "/persona" => {
                        let lib = match &persona_library {
                            Some(l) => l.clone(),
                            None => { eprintln!("(no --library configured)"); continue; }
                        };
                        if arg.is_empty() || arg == "list" {
                            eprintln!("(personas in {})", lib.display());
                            if let Ok(entries) = std::fs::read_dir(&lib) {
                                for ent in entries.flatten() {
                                    let p = ent.path();
                                    if p.extension().and_then(|s| s.to_str()) == Some("vec") {
                                        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                                            eprintln!("  {stem}");
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        if arg == "none" {
                            active_persona = None;
                            eprintln!("(persona cleared)");
                            continue;
                        }
                        // Load <name>.vec from the library.
                        let p = lib.join(format!("{arg}.vec"));
                        match load_persona_vec(&p, hidden_size) {
                            Ok(v) => {
                                let n2: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                                eprintln!("(persona {arg} loaded, ‖={:.4})", n2);
                                active_persona = Some((arg.to_string(), v));
                            }
                            Err(e) => eprintln!("(failed to load {}: {e})", p.display()),
                        }
                        continue;
                    }
                    "/introspect" => {
                        let h = match &last_response_hidden {
                            Some(h) => h,
                            None => { eprintln!("(no last response)"); continue; }
                        };
                        let h_norm: f32 = h.iter().map(|x| x*x).sum::<f32>().sqrt();

                        let mut entries: Vec<(String, f32, f32)> = Vec::new();
                        // Always include user_vec.
                        let user_n: f32 = user_vec.v.iter().map(|x| x*x).sum::<f32>().sqrt();
                        if user_n > 0.0 {
                            let dot: f32 = h.iter().zip(user_vec.v.iter()).map(|(a,b)| a*b).sum();
                            entries.push(("(user)".into(), dot, dot / (h_norm * user_n).max(1e-9)));
                        }
                        // All vectors in the library.
                        if let Some(lib) = &persona_library {
                            if let Ok(dir) = std::fs::read_dir(lib) {
                                for ent in dir.flatten() {
                                    let p = ent.path();
                                    if p.extension().and_then(|s| s.to_str()) != Some("vec") { continue; }
                                    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                                    if let Ok(v) = load_persona_vec(&p, hidden_size) {
                                        let vn: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                                        if vn == 0.0 { continue; }
                                        let dot: f32 = h.iter().zip(v.iter()).map(|(a,b)| a*b).sum();
                                        entries.push((stem, dot, dot / (h_norm * vn).max(1e-9)));
                                    }
                                }
                            }
                        }
                        entries.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
                        eprintln!("(activations on last response, ranked by |dot|):");
                        for (label, dot, cos) in entries.iter().take(10) {
                            eprintln!("  {label:<24} dot={dot:>+10.3}  cos={cos:>+6.3}");
                        }
                        continue;
                    }
                    "/+" | "/-" => {
                        if !steering_enabled { eprintln!("(no --user set; nothing to update)"); continue; }
                        let label: i8 = if cmd == "/+" { 1 } else { -1 };
                        match &last_response_hidden {
                            Some(h) => {
                                state.observe(label, h);
                                user_vec.v = state.vector();
                                user_vec.save(&vec_path).ok();
                                state.save(&state_path).ok();
                                eprintln!("({} → ‖v‖₂={:.4}, n_pos={}, n_neg={})",
                                    if label > 0 { "👍" } else { "👎" },
                                    user_vec.norm(), state.n_pos, state.n_neg);
                            }
                            None => eprintln!("(no last response to mark)"),
                        }
                        continue;
                    }
                    _ => { eprintln!("(unknown command: {cmd})"); continue; }
                }
            }

            // Capture the user-message hidden (alone, no template) for /topic.
            if let Ok(uids) = $pipeline.tokenizer.encode(input) {
                if !uids.is_empty() {
                    $pipeline.model.reset_kv_caches();
                    let h = $pipeline.model.forward_prefill_hidden(&uids);
                    user_msg_hiddens.push(h);
                    user_msg_texts.push(input.to_string());
                }
            }

            history.push(ChatMessage::user(input));
            let prompt = $template.apply(&history);

            // Probe-cache check: if the prompt's hidden has high cosine to a
            // dictionary entry, return the cached response instead of generating.
            if !probe_dict.is_empty() {
                let prompt_ids = $pipeline.tokenizer.encode(&prompt).unwrap_or_default();
                if !prompt_ids.is_empty() {
                    $pipeline.model.reset_kv_caches();
                    let h = $pipeline.model.forward_prefill_hidden(&prompt_ids);
                    let h_norm: f32 = h.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let mut best_idx = 0usize; let mut best_cos = f32::NEG_INFINITY;
                    for (i, (_, dh, dn)) in probe_dict.iter().enumerate() {
                        let dot: f32 = h.iter().zip(dh.iter()).map(|(a, b)| a*b).sum();
                        let denom = h_norm * dn;
                        let c = if denom == 0.0 { 0.0 } else { dot / denom };
                        if c > best_cos { best_cos = c; best_idx = i; }
                    }
                    if best_cos >= cache_threshold_f {
                        let cached = &probe_dict[best_idx].0;
                        eprintln!("(cache hit: cos={best_cos:.3} → {:?})",
                            cached.chars().take(60).collect::<String>());
                        println!("{cached}");
                        history.push(ChatMessage::assistant(cached.clone()));
                        last_response_hidden = Some(probe_dict[best_idx].1.clone());
                        continue;
                    }
                }
            }

            // Compose user_vec + active_persona into the steering vector for this turn.
            let mut combined_vec = user_vec.v.clone();
            if let Some((_, p)) = &active_persona {
                for (a, b) in combined_vec.iter_mut().zip(p.iter()) { *a += *b; }
            }

            let mut response = String::new();
            let result = if steering_enabled && alpha != 0.0 {
                $pipeline.generate_streaming_steered(
                    &prompt, &$config, &mut rng,
                    &combined_vec, alpha,
                    |token_text| {
                        print!("{token_text}");
                        io::stdout().flush().unwrap();
                        response.push_str(token_text);
                        true
                    },
                    |h| { last_response_hidden = Some(h.to_vec()); }
                )
            } else {
                $pipeline.generate_streaming(&prompt, &$config, &mut rng, |token_text| {
                    print!("{token_text}");
                    io::stdout().flush().unwrap();
                    response.push_str(token_text);
                    true
                })
            };

            match result {
                Ok(_) => {
                    println!();
                    history.push(ChatMessage::assistant(response));
                }
                Err(e) => eprintln!("\nGeneration error: {e}"),
            }
        }

        if steering_enabled {
            user_vec.save(&vec_path).ok();
            state.save(&state_path).ok();
            eprintln!("(saved {} and {})", vec_path.display(), state_path.display());
        }
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [--temp <f32>] [--top-k <n>] [--top-p <f32>] [--max-tokens <n>] [--template <name>] [--system <msg>] [--sparse] [--head-sparsity <f32>] [--neuron-sparsity <f32>]", args[0]);
        eprintln!();
        eprintln!("Templates: zephyr, chatml, llama2, llama3, mistral, raw, auto (default)");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);

    // Parse CLI args
    let mut temperature = 0.7f32;
    let mut top_k = 40usize;
    let mut top_p = 0.9f32;
    let mut max_tokens = 512usize;
    let mut rep_penalty = 1.1f32;
    let mut template_name = "auto".to_string();
    let mut system_msg: Option<String> = None;
    let mut use_sparse = false;
    let mut head_sparsity = 0.5f32;
    let mut neuron_sparsity = 0.5f32;
    let mut user_dir: Option<PathBuf> = None;
    let mut alpha: f32 = 1.0;
    let mut library: Option<PathBuf> = None;
    let mut calibration_name: Option<String> = None;
    let mut profiles_path: Option<PathBuf> = None;
    let mut cache_path: Option<PathBuf> = None;
    let mut cache_threshold: f32 = 0.95;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" | "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("Invalid temperature");
            }
            "--top-k" => {
                i += 1;
                top_k = args[i].parse().expect("Invalid top-k");
            }
            "--top-p" => {
                i += 1;
                top_p = args[i].parse().expect("Invalid top-p");
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("Invalid max-tokens");
            }
            "--rep-penalty" => {
                i += 1;
                rep_penalty = args[i].parse().expect("Invalid rep-penalty");
            }
            "--template" => {
                i += 1;
                template_name = args[i].clone();
            }
            "--system" => {
                i += 1;
                system_msg = Some(args[i].clone());
            }
            "--sparse" => {
                use_sparse = true;
            }
            "--head-sparsity" => {
                i += 1;
                head_sparsity = args[i].parse().expect("Invalid head-sparsity");
            }
            "--neuron-sparsity" => {
                i += 1;
                neuron_sparsity = args[i].parse().expect("Invalid neuron-sparsity");
            }
            "--user" => {
                i += 1;
                user_dir = Some(PathBuf::from(&args[i]));
            }
            "--alpha" => {
                i += 1;
                alpha = args[i].parse().expect("Invalid alpha");
            }
            "--library" => {
                i += 1;
                library = Some(PathBuf::from(&args[i]));
            }
            "--calibration" => {
                i += 1;
                calibration_name = Some(args[i].clone());
            }
            "--profiles" => {
                i += 1;
                profiles_path = Some(PathBuf::from(&args[i]));
            }
            "--cache" => {
                i += 1;
                cache_path = Some(PathBuf::from(&args[i]));
            }
            "--cache-threshold" => {
                i += 1;
                cache_threshold = args[i].parse().expect("Invalid cache-threshold");
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Detect or use specified template
    let template = match template_name.as_str() {
        "auto" => {
            let detected = ChatTemplate::detect(&model_dir).unwrap_or(ChatTemplate::Raw);
            eprintln!("Auto-detected template: {detected:?}");
            detected
        }
        "zephyr" => ChatTemplate::Zephyr,
        "chatml" => ChatTemplate::ChatML,
        "llama2" => ChatTemplate::Llama2,
        "llama3" => ChatTemplate::Llama3,
        "mistral" => ChatTemplate::MistralInstruct,
        "raw" => ChatTemplate::Raw,
        other => {
            eprintln!("Unknown template: {other}");
            std::process::exit(1);
        }
    };

    // Detect EOS token from model config
    let eos_token_ids = detect_eos_tokens(&model_dir);
    eprintln!("EOS token IDs: {eos_token_ids:?}");

    let config = GenerateConfig {
        max_new_tokens: max_tokens,
        sampler: SamplerConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: rep_penalty,
        },
        eos_token_ids,
    };

    if use_sparse {
        #[cfg(not(feature = "sparse"))]
        {
            eprintln!("Error: --sparse requires the 'sparse' feature. Rebuild with --features sparse");
            std::process::exit(1);
        }

        #[cfg(feature = "sparse")]
        {
            eprintln!(
                "Loading sparse model from {} (head_sparsity={head_sparsity}, neuron_sparsity={neuron_sparsity})...",
                model_dir.display()
            );
            let mut pipeline =
                SparsePipeline::from_dir(&model_dir, head_sparsity, neuron_sparsity)
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to load model: {e}");
                        std::process::exit(1);
                    });
            eprintln!(
                "Model loaded: {} layers, hidden_size={}, vocab_size={}",
                pipeline.model.model.config.num_layers,
                pipeline.model.model.config.hidden_size,
                pipeline.model.model.config.vocab_size
            );
            // Sparse pipeline path: steering not yet plumbed; fall back to plain chat.
            let user_state: Option<(PathBuf, PathBuf, UserVector, WelfordState, f32)> = None;
            let library: Option<PathBuf> = None;
            run_chat_loop!(pipeline, template, config, system_msg, user_state, library, calibration_name, profiles_path, cache_path, cache_threshold);
        }
    } else {
        eprintln!("Loading model from {}...", model_dir.display());
        let mut pipeline = Pipeline::from_dir(&model_dir).unwrap_or_else(|e| {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "Model loaded: {} layers, hidden_size={}, vocab_size={}",
            pipeline.model.config.num_layers,
            pipeline.model.config.hidden_size,
            pipeline.model.config.vocab_size
        );
        let hidden_size = pipeline.model.config.hidden_size;
        let user_state = user_dir.as_ref().map(|d| {
            std::fs::create_dir_all(d).ok();
            let vp = d.join("user.vec");
            let sp = d.join("user.state");
            // Detect model mismatch: if user.vec exists but its hidden_size
            // differs from the current model's, the vector was fit on a
            // different model. Warn and suggest vector_migrate.
            let v = if vp.exists() {
                match UserVector::load(&vp, hidden_size) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("⚠ user.vec at {} exists but doesn't match this model:", vp.display());
                        eprintln!("    {e}");
                        eprintln!("    To migrate across models, fit a kernel adapter from the");
                        eprintln!("    old model to this one (kernel_translate fit), then run:");
                        eprintln!("      vector_migrate --in {} --kernel A-to-B.kkernel \\", vp.display());
                        eprintln!("        --a-model OLD_MODEL --reference-corpus FILE \\");
                        eprintln!("        --out {}", vp.display());
                        eprintln!("    For now, starting fresh.");
                        UserVector::empty(hidden_size)
                    }
                }
            } else {
                UserVector::empty(hidden_size)
            };
            let s = WelfordState::load(&sp, hidden_size).unwrap_or_else(|_| WelfordState::empty(hidden_size));
            eprintln!("User vector loaded from {}: ‖v‖₂={:.4}, n_pos={}, n_neg={}, α={alpha}",
                d.display(), v.norm(), s.n_pos, s.n_neg);
            (vp, sp, v, s, alpha)
        });
        run_chat_loop!(pipeline, template, config, system_msg, user_state, library, calibration_name, profiles_path, cache_path, cache_threshold);
    }
}
