//! Read-only comparison of teacher forcing, cached logits and image completion.
use clap::Parser;
use klearu_image::{checkpoint::load_model, conditioning,
    sample::{build_prefix, sample_image_tokens_from_prefix, SampleConfig},
    train::{assemble_batch, read_manifest, TrainExample}};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)] checkpoint: PathBuf,
    #[arg(long)] manifest: PathBuf,
    #[arg(long)] out: PathBuf,
}
fn main() -> Result<(),Box<dyn std::error::Error+Send+Sync>> {
    let a=Args::parse();
    let model=load_model(&a.checkpoint)?;
    let tok=conditioning::resolve(&model.conditioning,None,false)?.ok_or("Conditional checkpoint required")?;
    let v=model.config.vocab_image;
    let n=model.config.image_grid_h*model.config.image_grid_w;
    let mut records=Vec::new();
    for (index,entry) in read_manifest(&a.manifest)?.into_iter().enumerate() {
        let text=conditioning::encode(&tok,&model.config,&entry.caption)?;
        let ex=TrainExample{text_tokens:text.clone(),image_tokens:entry.image_tokens.clone()};
        let batch=assemble_batch(&model.config,&ex)?;
        let dense=model.forward(&batch.token_ids)?;
        let prefix=build_prefix(&model.config,&text)?;
        assert_eq!(prefix,&batch.token_ids[..prefix.len()]);
        let mut session=model.decode_session();
        let mut cached=session.forward(&prefix)?;
        let mut max_error=0.0_f32;
        let mut squared_error=0.0_f64;
        let mut teacher_tokens=Vec::new();
        let mut nll=Vec::new();
        let mut entropy=Vec::new();
        let mut top64_mass=Vec::new();
        for (i,&pos) in batch.predict_at.iter().enumerate() {
            let row=&dense[pos*v..(pos+1)*v];
            for (&x,&y) in row.iter().zip(&cached) {
                if !x.is_finite() || !y.is_finite() { return Err("Non-finite logits".into()); }
                max_error=max_error.max((x-y).abs());
                squared_error+=((x-y) as f64).powi(2);
            }
            let id=row.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
            teacher_tokens.push(id as u32);
            let max=row[id] as f64;
            let weights:Vec<f64>=row.iter().map(|&x|(x as f64-max).exp()).collect();
            let sum: f64=weights.iter().sum();
            let mut probs:Vec<f64>=weights.iter().map(|x|x/sum).collect();
            nll.push(sum.ln()+max-row[batch.targets[i] as usize] as f64);
            entropy.push(-probs.iter().filter(|&&p|p>0.0).map(|p|p*p.ln()).sum::<f64>());
            probs.sort_unstable_by(|a,b|b.total_cmp(a));
            top64_mass.push(probs.iter().take(64).sum::<f64>());
            if i+1<n { cached=session.forward(&[model.image_token_to_id(batch.targets[i])])?; }
        }
        let mut completions=Vec::new();
        for length in [0,n/4,n/2,3*n/4] {
            for (mode,k,temp,p) in [("greedy",1,1.0,0.0),("standard",64,0.9,0.95),("wide",0,1.0,0.95)] {
                let cfg=SampleConfig{top_k:k,temperature:temp,top_p:p,seed:0};
                let tokens=sample_image_tokens_from_prefix(&model,&text,&batch.targets[..length],&cfg)?;
                assert_eq!(&tokens[..length],&batch.targets[..length]);
                completions.push(serde_json::json!({"mode":mode,"prefix_tokens":length,"image_tokens":tokens}));
            }
        }
        records.push(serde_json::json!({"index":index,"caption":entry.caption,"target_tokens":batch.targets,
            "teacher_argmax_tokens":teacher_tokens,"teacher_nll":nll,"teacher_entropy":entropy,
            "teacher_top64_mass":top64_mass,"cached_dense_max_abs_error":max_error,
            "cached_dense_rmse":(squared_error/(n*v) as f64).sqrt(),"completions":completions}));
        eprintln!("Completed row {index}, cache/dense max error={max_error}");
    }
    std::fs::write(a.out,serde_json::to_vec_pretty(&serde_json::json!({"records":records,"vocab_image":v,"image_tokens":n}))?)?;
    Ok(())
}
