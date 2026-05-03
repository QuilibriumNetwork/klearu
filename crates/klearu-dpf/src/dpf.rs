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

        // Correction word seed: XOR of the "lose" children. When applied via XOR
        // (gated on t[b]), the lose-side seeds become equal across parties.
        let mut cw_seed = [0u8; 16];
        for j in 0..16 {
            cw_seed[j] = s_children[0][lose][j] ^ s_children[1][lose][j];
        }

        // Correction word control bits.
        // After applying CW to party b's child (gated on t[b]):
        //   t'[b][side] = t_children[b][side] XOR (t[b] AND tcw[side])
        // Since the invariant t[0] XOR t[1] = 1 holds on the keep path:
        //   t'[0][side] XOR t'[1][side]
        //     = t_children[0][side] XOR t_children[1][side] XOR tcw[side]
        //
        // Goal on the LOSE side: s'0 = s'1 AND t'0 = t'1 so that subtree contributions
        // cancel under the (-1)^party output sign.
        //   tcw[lose] = t_children[0][lose] XOR t_children[1][lose]
        //
        // Goal on the KEEP side: maintain t'0 XOR t'1 = 1 so the recursion continues
        // to the target leaf with diverged state.
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
