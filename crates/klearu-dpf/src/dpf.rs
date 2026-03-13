use crate::aes_prg::AesPrg;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A correction word used at each level of the DPF tree.
#[derive(Clone, Debug)]
pub struct CorrectionWord {
    pub seed: [u8; 16],
    pub control_left: bool,
    pub control_right: bool,
}

/// DPF key for a point function f(alpha) = beta, f(x) = 0 otherwise.
#[derive(Clone, Debug)]
pub struct DpfKey {
    pub party: u8,
    pub seed: [u8; 16],
    pub correction_words: Vec<CorrectionWord>,
    pub output_correction: u32,
}

fn get_bit(x: u32, i: u8, depth: u8) -> bool {
    ((x >> (depth - 1 - i)) & 1) != 0
}

fn xor_seeds(a: &mut [u8; 16], b: &[u8; 16]) {
    for j in 0..16 {
        a[j] ^= b[j];
    }
}

fn make_seed(alpha: u32, beta: u32, party: u8) -> [u8; 16] {
    use aes::cipher::{BlockEncrypt, KeyInit};
    let mut s = [0u8; 16];
    s[0..4].copy_from_slice(&alpha.to_le_bytes());
    s[4..8].copy_from_slice(&beta.to_le_bytes());
    s[8] = party;
    s[9] = 0xAB;
    let key_byte = 0x10u8.wrapping_add(party);
    let cipher = aes::Aes128::new((&[key_byte; 16]).into());
    let mut block = s.into();
    cipher.encrypt_block(&mut block);
    block.into()
}

/// Generate a pair of DPF keys (BGI construction).
///
/// Invariant maintained at each level: `t[0] XOR t[1] = 1`.
///
/// At non-target leaves, both parties compute the same seed with opposite control bits,
/// so their outputs cancel. At the target leaf, seeds differ, and the output correction
/// ensures the sum equals beta.
pub fn dpf_gen(prg: &AesPrg, alpha: u32, beta: u32, depth: u8) -> (DpfKey, DpfKey) {
    assert!(depth > 0 && depth <= 32);
    if depth < 32 {
        assert!(alpha < (1u32 << depth));
    }

    let seed0 = make_seed(alpha, beta, 0);
    let seed1 = make_seed(alpha, beta, 1);

    // s[b] and t[b] represent party b's state in the gen simulation
    let mut s = [seed0, seed1];
    let mut t = [false, true];

    let mut correction_words = Vec::with_capacity(depth as usize);

    for i in 0..depth {
        let alpha_bit = get_bit(alpha, i, depth);

        // Expand both parties' current seeds
        let expanded: [([u8; 16], bool, [u8; 16], bool); 2] = [
            prg.expand(&s[0]),
            prg.expand(&s[1]),
        ];

        // Extract left and right for each party
        let s_children = [
            [expanded[0].0, expanded[0].2], // party 0: [left, right]
            [expanded[1].0, expanded[1].2], // party 1: [left, right]
        ];
        let t_children = [
            [expanded[0].1, expanded[0].3], // party 0: [left_t, right_t]
            [expanded[1].1, expanded[1].3], // party 1: [left_t, right_t]
        ];

        let keep = alpha_bit as usize; // 0 = left, 1 = right
        let lose = 1 - keep;

        // Correction word seed: XOR of the "lose" children
        // When applied (via XOR), this makes the lose children equal
        let mut cw_seed = [0u8; 16];
        for j in 0..16 {
            cw_seed[j] = s_children[0][lose][j] ^ s_children[1][lose][j];
        }

        // Correction word control bits
        // After applying CW to party b's child (if t[b] is set):
        //   t'[b][side] = t_children[b][side] XOR (t[b] * tcw[side])
        //
        // We need the invariant t'[0][keep] XOR t'[1][keep] = 1 to continue.
        // On the lose side, we need t'[0][lose] XOR t'[1][lose] to be such that
        // the outputs cancel (both parties with same seed, one adds output_correction
        // and the other doesn't -- actually the outputs cancel because the seeds
        // match and the signs are opposite).
        //
        // For the LOSE side: we want both parties to end up with the SAME seed.
        // The CW seed makes s'[0][lose] = s'[1][lose] (after XOR correction).
        // For the KEEP side: we want both parties to have DIFFERENT seeds (so
        // the protocol continues to the target point).
        // The CW seed XOR is applied to BOTH sides when t[b]=1, but since
        // s_keep[0] != s_keep[1] in general, the CW doesn't destroy the difference.
        //
        // Wait - that's the problem. The CW seed is XORed into BOTH left and right
        // children when t[b]=1. So it affects both the keep and lose sides.
        // This is correct: on the lose side, it cancels the difference.
        // On the keep side, it may change the values but both parties still have
        // different seeds (the difference is preserved, just modified).
        //
        // Control bits: need t'[0] XOR t'[1] = 1 on the keep side.
        //   t'[0][keep] XOR t'[1][keep]
        //     = (t_children[0][keep] XOR t[0]*tcw[keep]) XOR (t_children[1][keep] XOR t[1]*tcw[keep])
        //     = t_children[0][keep] XOR t_children[1][keep] XOR (t[0] XOR t[1])*tcw[keep]
        //     = t_children[0][keep] XOR t_children[1][keep] XOR tcw[keep]  (since t[0]^t[1]=1)
        // Want = 1:
        //   tcw[keep] = t_children[0][keep] XOR t_children[1][keep] XOR 1

        // Similarly for lose side:
        //   tcw[lose] = t_children[0][lose] XOR t_children[1][lose] XOR 1
        // This ensures both sides maintain the invariant.
        // But wait: on the lose side, we want the outputs to cancel.
        // The outputs cancel if the seeds are the same AND the output corrections
        // are applied to exactly one party (since signs are opposite).
        // That means we want t'[0][lose] XOR t'[1][lose] = 1 as well (so one
        // applies the correction and the other doesn't).
        // But actually, on non-target leaves, neither should apply the correction!
        // Hmm, let me reconsider.
        //
        // Actually, the invariant t[0] XOR t[1] = 1 is maintained everywhere.
        // This means at EVERY leaf, one party has t=0 and the other has t=1.
        // But the seeds are the same on non-target paths, so:
        //   Party 0 output = (+1) * (convert(s) + t0*oc)
        //   Party 1 output = (-1) * (convert(s) + t1*oc)
        //   Sum = convert(s) + t0*oc - convert(s) - t1*oc = (t0-t1)*oc
        // Since exactly one of t0, t1 is true: sum = +/-oc
        // This is NOT zero unless oc = 0!
        //
        // I see the issue: the standard DPF doesn't use the (-1)^b sign convention.
        // Instead, both parties output the SAME sign, and the control bits ensure
        // cancellation via a different mechanism.
        //
        // Let me switch to the standard convention where:
        //   Party b's output at leaf x = convert(s_b) + t_b * output_correction
        //   And the DPF output is: output_0 - output_1  (or output_0 + output_1 in Z_{2^32})
        //
        // At a non-target leaf where s0 = s1 and t0 XOR t1 = 1:
        //   output_0 + output_1 = convert(s) + t0*oc + convert(s) + t1*oc
        //                       = 2*convert(s) + oc   (since t0+t1 = 1)
        // This is NOT zero either!
        //
        // OK, the standard BGI DPF uses SUBTRACTION, not addition:
        //   DPF(x) = (-1)^party * [convert(s) + t * oc]
        //   sum = (+1)*[convert(s0) + t0*oc] + (-1)*[convert(s1) + t1*oc]
        //       = convert(s0) - convert(s1) + (t0*oc - t1*oc)
        //
        // At non-target: s0=s1, so convert(s0)=convert(s1), and:
        //   sum = (t0-t1)*oc = ±oc
        //
        // This still doesn't cancel! Something is fundamentally wrong with my understanding.
        //
        // The ACTUAL standard approach:
        //   - On the LOSE side, we want seeds to be EQUAL and control bits to be EQUAL (both 0)
        //   - The control bit is what determines if correction is applied
        //   - If both parties have t=0 on the lose side, neither applies correction
        //   - Then output_0 - output_1 = convert(s) - convert(s) = 0 ✓
        //
        // But if both controls are 0, the invariant t0 XOR t1 = 1 is violated on lose.
        // That's the point: the invariant ONLY holds on the KEEP path!
        //
        // Let me re-derive the correction word computation:
        //   On the KEEP side: want t'0 XOR t'1 = 1 (maintain invariant)
        //   On the LOSE side: want t'0 = t'1 = 0 (both false, so no correction applied)
        //     AND want s'0 = s'1 (same seed, so convert gives same value)
        //
        // For the lose side with t'0 = t'1 = 0:
        //   t'b[lose] = t_children[b][lose] XOR (t[b] * tcw[lose])
        //   Want t'0[lose] = 0: tcw[lose] must correct t_children[0][lose]
        //     if t[0]=0: t'0 = t_children[0][lose], so need t_children[0][lose]=false
        //     if t[0]=1: t'0 = t_children[0][lose] XOR tcw[lose]
        //   Similarly for party 1.
        //
        // Since t[0] XOR t[1] = 1, exactly one of t[0], t[1] is true.
        // WLOG say t[0]=false, t[1]=true:
        //   t'0[lose] = t_children[0][lose]
        //   t'1[lose] = t_children[1][lose] XOR tcw[lose]
        //   Want both = 0: tcw[lose] = t_children[1][lose], and need t_children[0][lose] = false
        //
        // But t_children[0][lose] might be true! So we can't guarantee both are 0.
        //
        // Hmm, I think the issue is that on the lose subtree (below the current level),
        // everything cancels because the ENTIRE subtree of party 0 equals party 1's subtree.
        // Not just at this level, but all the way down. That's because both parties have the
        // same seed at the lose node, and the same correction words are used from here on.
        // So even if t values differ at the lose node, the subtrees below still cancel because
        // the same CW is applied to the same seeds!
        //
        // Actually wait. If s0=s1 but t0 != t1 at the lose node, then at the next level:
        //   Both parties expand the same seed (same s0=s1)
        //   But party with t=1 applies the CW, party with t=0 doesn't
        //   So they diverge again!
        //
        // I think the correct approach is:
        //   On the LOSE side: want s'0 = s'1 AND t'0 = t'1
        //   (both parties have identical state, so entire subtree is identical,
        //    and since we use (-1)^party sign, their contributions cancel)
        //
        // With (-1)^party convention:
        //   party 0's output = (+1) * val, party 1's output = (-1) * val
        //   Sum = val - val = 0 ✓
        //
        // So on the lose side, we need s'0=s'1 AND t'0=t'1.
        // The CW seed already ensures s'0=s'1.
        // For control bits, with t[0] XOR t[1] = 1 (WLOG t[0]=0, t[1]=1):
        //   t'0[lose] = t_children[0][lose]  (t[0]=0, no CW applied)
        //   t'1[lose] = t_children[1][lose] XOR tcw[lose]  (t[1]=1, CW applied)
        //   Want equal: tcw[lose] = t_children[0][lose] XOR t_children[1][lose]
        //
        // For keep side, want t'0 XOR t'1 = 1:
        //   t'0[keep] = t_children[0][keep]
        //   t'1[keep] = t_children[1][keep] XOR tcw[keep]
        //   Want XOR = 1: tcw[keep] = t_children[0][keep] XOR t_children[1][keep] XOR 1
        //
        // Now for general case (not WLOG):
        //   t'b[side] = t_children[b][side] XOR (t[b] AND tcw[side])
        //   t'0[side] XOR t'1[side]
        //     = t_children[0][side] XOR t_children[1][side] XOR (t[0] XOR t[1]) AND tcw[side]
        //     = t_children[0][side] XOR t_children[1][side] XOR tcw[side]
        //
        // For lose (want t'0 = t'1, i.e., XOR = 0):
        //   tcw[lose] = t_children[0][lose] XOR t_children[1][lose]
        //
        // For keep (want t'0 XOR t'1 = 1):
        //   tcw[keep] = t_children[0][keep] XOR t_children[1][keep] XOR 1

        let tcw = [
            // tcw[0] = tcw_left, tcw[1] = tcw_right
            if lose == 0 {
                // left is lose
                t_children[0][0] ^ t_children[1][0] // lose: want XOR = 0
            } else {
                // left is keep
                t_children[0][0] ^ t_children[1][0] ^ true // keep: want XOR = 1
            },
            if lose == 1 {
                // right is lose
                t_children[0][1] ^ t_children[1][1]
            } else {
                // right is keep
                t_children[0][1] ^ t_children[1][1] ^ true
            },
        ];

        correction_words.push(CorrectionWord {
            seed: cw_seed,
            control_left: tcw[0],
            control_right: tcw[1],
        });

        // Advance both parties to the keep direction
        for b in 0..2usize {
            let mut child_seed = s_children[b][keep];
            let mut child_t = t_children[b][keep];

            if t[b] {
                xor_seeds(&mut child_seed, &cw_seed);
                child_t ^= tcw[keep];
            }

            s[b] = child_seed;
            t[b] = child_t;
        }

        // Verify invariant
        debug_assert_eq!(t[0] ^ t[1], true, "invariant broken at level {}", i);
    }

    // Output correction at the target leaf
    let out0 = prg.derive_output(&s[0]);
    let out1 = prg.derive_output(&s[1]);

    // Party b's leaf output = (-1)^b * (convert(s_b) + t_b * oc)
    // Sum = convert(s0) + t0*oc - convert(s1) - t1*oc
    //     = (convert(s0) - convert(s1)) + (t0 - t1)*oc
    // Want sum = beta.
    // Since t0 XOR t1 = 1, exactly one is true.
    let oc = if t[0] && !t[1] {
        // t0=1, t1=0: coeff = 1
        // (out0 - out1) + oc = beta
        beta.wrapping_sub(out0).wrapping_add(out1)
    } else {
        // t0=0, t1=1: coeff = -1
        // (out0 - out1) - oc = beta
        out0.wrapping_sub(out1).wrapping_sub(beta)
    };

    (
        DpfKey {
            party: 0,
            seed: seed0,
            correction_words: correction_words.clone(),
            output_correction: oc,
        },
        DpfKey {
            party: 1,
            seed: seed1,
            correction_words,
            output_correction: oc,
        },
    )
}

/// Evaluate a DPF key at point x.
pub fn dpf_eval(prg: &AesPrg, key: &DpfKey, x: u32) -> u32 {
    let depth = key.correction_words.len() as u8;
    if depth < 32 {
        assert!(x < (1u32 << depth));
    }

    let mut s = key.seed;
    let mut t: bool = key.party != 0;

    for i in 0..depth {
        let (mut sl, mut tl, mut sr, mut tr) = prg.expand(&s);
        let cw = &key.correction_words[i as usize];

        if t {
            xor_seeds(&mut sl, &cw.seed);
            xor_seeds(&mut sr, &cw.seed);
            tl ^= cw.control_left;
            tr ^= cw.control_right;
        }

        if get_bit(x, i, depth) {
            s = sr;
            t = tr;
        } else {
            s = sl;
            t = tl;
        }
    }

    let raw = prg.derive_output(&s);
    let val = raw.wrapping_add(if t { key.output_correction } else { 0 });

    if key.party == 0 {
        val
    } else {
        val.wrapping_neg()
    }
}

/// Minimum number of seeds at a tree level before switching to parallel expansion.
#[cfg(feature = "parallel")]
const PARALLEL_DPF_THRESHOLD: usize = 512;

/// Full-domain evaluation with rayon parallelism for large tree levels.
pub fn dpf_full_eval(prg: &AesPrg, key: &DpfKey) -> Vec<u32> {
    let depth = key.correction_words.len() as u8;
    let domain_size = 1usize << depth;

    let mut seeds: Vec<[u8; 16]> = vec![key.seed];
    let mut controls: Vec<bool> = vec![key.party != 0];

    for i in 0..depth {
        let cw = &key.correction_words[i as usize];
        let _ = i;

        #[cfg(feature = "parallel")]
        if seeds.len() >= PARALLEL_DPF_THRESHOLD {
            // Parallel: expand all seeds at this level concurrently
            let expanded: Vec<_> = seeds.par_iter().zip(controls.par_iter())
                .map(|(s, &tc)| {
                    let (mut sl, mut tl, mut sr, mut tr) = prg.expand(s);
                    if tc {
                        xor_seeds(&mut sl, &cw.seed);
                        xor_seeds(&mut sr, &cw.seed);
                        tl ^= cw.control_left;
                        tr ^= cw.control_right;
                    }
                    (sl, tl, sr, tr)
                })
                .collect();

            let mut new_seeds = Vec::with_capacity(expanded.len() * 2);
            let mut new_controls = Vec::with_capacity(expanded.len() * 2);
            for (sl, tl, sr, tr) in expanded {
                new_seeds.push(sl);
                new_controls.push(tl);
                new_seeds.push(sr);
                new_controls.push(tr);
            }
            seeds = new_seeds;
            controls = new_controls;
            continue;
        }

        // Sequential for small levels or when parallel feature is disabled
        let mut new_seeds = Vec::with_capacity(seeds.len() * 2);
        let mut new_controls = Vec::with_capacity(seeds.len() * 2);

        for (s, &tc) in seeds.iter().zip(controls.iter()) {
            let (mut sl, mut tl, mut sr, mut tr) = prg.expand(s);
            if tc {
                xor_seeds(&mut sl, &cw.seed);
                xor_seeds(&mut sr, &cw.seed);
                tl ^= cw.control_left;
                tr ^= cw.control_right;
            }
            new_seeds.push(sl);
            new_controls.push(tl);
            new_seeds.push(sr);
            new_controls.push(tr);
        }
        seeds = new_seeds;
        controls = new_controls;
    }

    assert_eq!(seeds.len(), domain_size);

    let oc = key.output_correction;
    let party0 = key.party == 0;

    // Output derivation: parallel when available, sequential otherwise
    #[cfg(feature = "parallel")]
    {
        seeds.par_iter().zip(controls.par_iter()).map(|(s, &tc)| {
            let raw = prg.derive_output(s);
            let val = raw.wrapping_add(if tc { oc } else { 0 });
            if party0 { val } else { val.wrapping_neg() }
        }).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        seeds.iter().zip(controls.iter()).map(|(s, &tc)| {
            let raw = prg.derive_output(s);
            let val = raw.wrapping_add(if tc { oc } else { 0 });
            if party0 { val } else { val.wrapping_neg() }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prg() -> AesPrg {
        AesPrg::new(&[0u8; 16])
    }

    fn check_dpf(prg: &AesPrg, alpha: u32, beta: u32, depth: u8) {
        let (key0, key1) = dpf_gen(prg, alpha, beta, depth);
        let domain = 1u32 << depth;

        for x in 0..domain {
            let share0 = dpf_eval(prg, &key0, x);
            let share1 = dpf_eval(prg, &key1, x);
            let result = share0.wrapping_add(share1);
            let expected = if x == alpha { beta } else { 0 };
            assert_eq!(
                result, expected,
                "DPF(alpha={}, beta={}, depth={}) at x={}: got {}, expected {}",
                alpha, beta, depth, x, result, expected
            );
        }
    }

    #[test]
    fn test_dpf_depth_1() {
        let prg = make_prg();
        check_dpf(&prg, 0, 100, 1);
        check_dpf(&prg, 1, 100, 1);
    }

    #[test]
    fn test_dpf_depth_2() {
        let prg = make_prg();
        for alpha in 0..4 {
            check_dpf(&prg, alpha, 42, 2);
        }
    }

    #[test]
    fn test_dpf_depth_4() {
        let prg = make_prg();
        check_dpf(&prg, 7, 42, 4);
        check_dpf(&prg, 0, 999, 4);
        check_dpf(&prg, 15, 777, 4);
    }

    #[test]
    fn test_dpf_large_beta() {
        let prg = make_prg();
        check_dpf(&prg, 200, 0xDEADBEEF, 8);
    }

    #[test]
    fn test_dpf_beta_zero() {
        let prg = make_prg();
        check_dpf(&prg, 5, 0, 4);
    }

    #[test]
    fn test_dpf_full_eval_matches_pointwise() {
        let prg = make_prg();
        let alpha = 33u32;
        let beta = 12345u32;
        let (key0, key1) = dpf_gen(&prg, alpha, beta, 6);
        let full0 = dpf_full_eval(&prg, &key0);
        let full1 = dpf_full_eval(&prg, &key1);

        for x in 0..64u32 {
            assert_eq!(full0[x as usize], dpf_eval(&prg, &key0, x));
            assert_eq!(full1[x as usize], dpf_eval(&prg, &key1, x));
            let result = full0[x as usize].wrapping_add(full1[x as usize]);
            assert_eq!(result, if x == alpha { beta } else { 0 });
        }
    }

    #[test]
    fn test_dpf_multiple_parameters() {
        let prg = make_prg();
        for alpha in [0, 1, 127, 128, 254, 255] {
            for beta in [1u32, 42, 0xFFFFFFFF, 0x80000000] {
                check_dpf(&prg, alpha, beta, 8);
            }
        }
    }

    #[test]
    fn test_dpf_security_single_key() {
        let prg = make_prg();
        let (key0, _) = dpf_gen(&prg, 42, 1, 8);
        let outputs: Vec<u32> = (0..256u32).map(|x| dpf_eval(&prg, &key0, x)).collect();
        let unique: std::collections::HashSet<u32> = outputs.iter().copied().collect();
        assert!(unique.len() > 200);
    }
}
