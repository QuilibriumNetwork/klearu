/// Head-level sparsity for attention.
///
/// Given predicted head importance scores, select the top-k heads and only
/// compute Q/K/V and attention for those heads. Inactive head positions are
/// zeroed in the concatenated output before the output projection.
pub fn select_active_heads(importance_scores: &[f32], num_heads: usize, sparsity: f32) -> Vec<usize> {
    let k = ((num_heads as f32 * sparsity).ceil() as usize).max(1).min(num_heads);

    let mut indexed: Vec<(usize, f32)> = importance_scores
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
    selected.sort_unstable();
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_all_heads() {
        let scores = vec![1.0; 4];
        let selected = select_active_heads(&scores, 4, 1.0);
        assert_eq!(selected, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_select_top_half() {
        let scores = vec![0.1, 0.9, 0.3, 0.8];
        let selected = select_active_heads(&scores, 4, 0.5);
        assert_eq!(selected, vec![1, 3]);
    }

    #[test]
    fn test_select_minimum_one() {
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let selected = select_active_heads(&scores, 4, 0.0);
        assert_eq!(selected.len(), 1);
    }
}
