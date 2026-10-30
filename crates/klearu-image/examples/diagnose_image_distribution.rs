//! Frozen-checkpoint diagnostic: probability-weighted latent means.
use clap::Parser;
use klearu_image::{checkpoint::load_model, conditioning,
    sample::build_prefix, train::{assemble_batch,read_manifest,TrainExample}};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)] checkpoint: PathBuf,
    #[arg(long)] manifest: PathBuf,
    #[arg(long)] codebook_json: PathBuf,
    #[arg(long)] out: PathBuf,
}
fn moments(row:&[f32],cb:&[[f64;4]]) -> ([f64;4],usize,usize) {
    let argmax=row.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
    let max=row[argmax] as f64;
    let mut mean=[0.0;4]; let mut total=0.0;
    for (&logit,point) in row.iter().zip(cb) {
        let w=(logit as f64-max).exp(); total+=w;
        for c in 0..4 { mean[c]+=w*point[c]; }
    }
    for x in &mut mean { *x/=total; }
    let nearest=cb.iter().enumerate().min_by(|a,b| {
        let dist=|p:&[f64;4]|(0..4).map(|c|(mean[c]-p[c]).powi(2)).sum::<f64>();
        dist(a.1).total_cmp(&dist(b.1))
    }).unwrap().0;
    (mean,argmax,nearest)
}
fn main()->Result<(),Box<dyn std::error::Error+Send+Sync>> {
    let a=Args::parse(); let model=load_model(&a.checkpoint)?;
    let cb:Vec<[f64;4]>=serde_json::from_slice(&std::fs::read(a.codebook_json)?)?;
    let v=model.config.vocab_image;
    if cb.len()!=v || cb.iter().flatten().any(|x|!x.is_finite()) { return Err("Invalid codebook".into()); }
    let tok=conditioning::resolve(&model.conditioning,None,false)?.ok_or("Conditional checkpoint required")?;
    let data=read_manifest(&a.manifest)?;
    let texts:Vec<Vec<u32>>=data.iter().map(|r|conditioning::encode(&tok,&model.config,&r.caption)).collect::<Result<_,_>>()?;
    let mut records=Vec::new();
    for (index,entry) in data.iter().enumerate() {
        let mut teacher=Vec::new();
        for (mode,text) in [("correct",texts[index].clone()),("shuffled",texts[(index+1)%texts.len()].clone()),("empty",vec![])] {
            let batch=assemble_batch(&model.config,&TrainExample{text_tokens:text,image_tokens:entry.image_tokens.clone()})?;
            let logits=model.forward(&batch.token_ids)?;
            let mut means=Vec::new(); let mut ids=Vec::new(); let mut nearest=Vec::new();
            for &pos in &batch.predict_at {
                let (mean,id,near)=moments(&logits[pos*v..(pos+1)*v],&cb);
                means.push(mean); ids.push(id); nearest.push(near);
            }
            teacher.push(serde_json::json!({"mode":mode,"means":means,"argmax_tokens":ids,"nearest_mean_tokens":nearest}));
        }
        let n=entry.image_tokens.len(); let mut completions=Vec::new();
        for length in [0,n/2] {
            let mut seq=build_prefix(&model.config,&texts[index])?;
            seq.extend(entry.image_tokens[..length].iter().map(|&id|model.image_token_to_id(id)));
            let mut session=model.decode_session(); let mut logits=session.forward(&seq)?;
            let mut ids=entry.image_tokens[..length].to_vec();
            for i in length..n {
                let (_,_,id)=moments(&logits,&cb); ids.push(id as u32);
                if i+1<n { logits=session.forward(&[model.image_token_to_id(id as u32)])?; }
            }
            completions.push(serde_json::json!({"prefix_tokens":length,"image_tokens":ids}));
        }
        records.push(serde_json::json!({"caption":entry.caption,"target_tokens":entry.image_tokens,"teacher":teacher,"nearest_mean_completions":completions}));
        eprintln!("Completed row {index}");
    }
    std::fs::write(a.out,serde_json::to_vec_pretty(&serde_json::json!({"records":records}))?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mean_and_nearest_match_hand_calculation() {
        let cb=[[0.0;4],[2.0;4],[1.0;4]];
        let (mean,argmax,near)=moments(&[0.0,0.0,-100.0],&cb);
        assert!(mean.iter().all(|x|(x-1.0).abs()<1e-12));
        assert!(argmax<2); assert_eq!(near,2);
        let (mean,_,near)=moments(&[-1000.0,0.0,-1000.0],&cb);
        assert_eq!(mean,[2.0;4]); assert_eq!(near,1);
    }
}
