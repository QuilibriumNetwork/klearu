use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use klearu_core::config::*;
use klearu_core::data::Example;
use klearu_core::hash::{HashFamily, MinHash, SimHash, SparseRandomProjection, WtaHash};
use klearu_core::lsh::{create_lsh_index, LshIndexTrait};
use klearu_core::network::Network;
use klearu_core::optim::HogwildNetwork;
use klearu_core::tensor::SparseVector;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_dense(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn sparse_dense(seed: u64, dim: usize, density: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim)
        .map(|_| {
            if rng.gen::<f32>() < density {
                rng.gen_range(-1.0..1.0)
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Hash Function Throughput
// ---------------------------------------------------------------------------

fn bench_hash_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_throughput");

    const K: usize = 8;
    const L: usize = 4;
    const SEED: u64 = 42;

    // SimHash dim=128
    {
        let dim = 128;
        let hasher = SimHash::new(dim, K, L, SEED);
        let input = random_dense(100, dim);
        group.bench_function("simhash_dim128", |b| {
            b.iter(|| {
                for t in 0..L {
                    black_box(hasher.hash_dense(black_box(&input), t));
                }
            })
        });
    }

    // SimHash dim=1024
    {
        let dim = 1024;
        let hasher = SimHash::new(dim, K, L, SEED);
        let input = random_dense(101, dim);
        group.bench_function("simhash_dim1024", |b| {
            b.iter(|| {
                for t in 0..L {
                    black_box(hasher.hash_dense(black_box(&input), t));
                }
            })
        });
    }

    // WtaHash dim=128
    {
        let dim = 128;
        let window_size = 8;
        let hasher = WtaHash::new(dim, K, L, window_size, SEED);
        let input = random_dense(102, dim);
        group.bench_function("wtahash_dim128", |b| {
            b.iter(|| {
                for t in 0..L {
                    black_box(hasher.hash_dense(black_box(&input), t));
                }
            })
        });
    }

    // MinHash dim=128
    {
        let dim = 128;
        let hasher = MinHash::new(dim, K, L, SEED);
        let input = random_dense(103, dim);
        group.bench_function("minhash_dim128", |b| {
            b.iter(|| {
                for t in 0..L {
                    black_box(hasher.hash_dense(black_box(&input), t));
                }
            })
        });
    }

    // SparseRandomProjection dim=128
    {
        let dim = 128;
        let sparsity = 1.0 / 3.0;
        let hasher = SparseRandomProjection::new(dim, K, L, sparsity, SEED);
        let input = random_dense(104, dim);
        group.bench_function("srp_dim128", |b| {
            b.iter(|| {
                for t in 0..L {
                    black_box(hasher.hash_dense(black_box(&input), t));
                }
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. LSH Index Operations
// ---------------------------------------------------------------------------

fn build_populated_index(dim: usize, n_neurons: usize) -> (Box<dyn LshIndexTrait>, Vec<Vec<f32>>) {
    let config = LshConfig {
        hash_function: HashFunctionType::SimHash,
        bucket_type: BucketType::Fifo,
        num_tables: 50,
        range_pow: 6,
        num_hashes: 6,
        bucket_capacity: 128,
        rebuild_interval_base: 100,
        rebuild_decay: 0.1,
    };

    let mut index = create_lsh_index(&config, dim, 42);

    let mut rng = StdRng::seed_from_u64(123);
    let mut weight_vecs = Vec::with_capacity(n_neurons);
    for id in 0..n_neurons as u32 {
        let weights: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        index.insert(id, &weights);
        weight_vecs.push(weights);
    }

    (index, weight_vecs)
}

fn bench_lsh_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_index");

    let dim = 128;
    let n_neurons = 1000;

    // Benchmark building + inserting 1000 neurons
    group.bench_function("create_and_insert_1000", |b| {
        b.iter(|| {
            let (index, _) = build_populated_index(black_box(dim), black_box(n_neurons));
            black_box(index);
        })
    });

    // Benchmark query_union on a populated index
    {
        let (index, _) = build_populated_index(dim, n_neurons);
        let query = random_dense(999, dim);
        group.bench_function("query_union_1000", |b| {
            b.iter(|| {
                black_box(index.query_union(black_box(&query)));
            })
        });
    }

    // Benchmark query_with_counts on a populated index
    {
        let (index, _) = build_populated_index(dim, n_neurons);
        let query = random_dense(998, dim);
        group.bench_function("query_with_counts_1000", |b| {
            b.iter(|| {
                black_box(index.query_with_counts(black_box(&query)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Sparse Vector Operations
// ---------------------------------------------------------------------------

fn bench_sparse_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector");

    let dim = 1024;
    let density = 0.1; // 10% sparsity

    // dot_dense
    {
        let dense_data = sparse_dense(200, dim, density);
        let sv = SparseVector::from_dense(&dense_data);
        let other_dense = random_dense(201, dim);
        group.bench_function("dot_dense_dim1024_10pct", |b| {
            b.iter(|| {
                black_box(sv.dot_dense(black_box(&other_dense)));
            })
        });
    }

    // from_dense
    {
        let dense_data = sparse_dense(202, dim, density);
        group.bench_function("from_dense_dim1024", |b| {
            b.iter(|| {
                black_box(SparseVector::from_dense(black_box(&dense_data)));
            })
        });
    }

    // to_dense
    {
        let dense_data = sparse_dense(203, dim, density);
        let sv = SparseVector::from_dense(&dense_data);
        group.bench_function("to_dense_dim1024", |b| {
            b.iter(|| {
                black_box(sv.to_dense());
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Forward Pass
// ---------------------------------------------------------------------------

fn make_bench_config() -> SlideConfig {
    SlideConfig {
        network: NetworkConfig {
            layers: vec![
                LayerConfig {
                    input_dim: 128,
                    num_neurons: 64,
                    activation: ActivationType::Relu,
                    lsh: LshConfig {
                        num_tables: 10,
                        num_hashes: 6,
                        ..LshConfig::default()
                    },
                    sampling: SamplingType::TopK,
                    sampling_threshold: 1,
                    top_k: 64, // activate all for deterministic timing
                    is_output: false,
                },
                LayerConfig {
                    input_dim: 64,
                    num_neurons: 10,
                    activation: ActivationType::Softmax,
                    lsh: LshConfig {
                        num_tables: 10,
                        num_hashes: 6,
                        ..LshConfig::default()
                    },
                    sampling: SamplingType::TopK,
                    sampling_threshold: 1,
                    top_k: 10, // activate all for deterministic timing
                    is_output: true,
                },
            ],
            optimizer: OptimizerType::Sgd,
            learning_rate: 0.01,
            batch_size: 1,
            num_threads: 1,
        },
        seed: 42,
        hogwild: false,
    }
}

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");

    let config = make_bench_config();
    let network = Network::new(config);
    let input = random_dense(300, 128);

    group.bench_function("forward_128_64_10", |b| {
        b.iter(|| {
            black_box(network.forward(black_box(&input)));
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Train Step
// ---------------------------------------------------------------------------

fn bench_train_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_step");

    let config = make_bench_config();
    let mut network = Network::new(config);
    let example = Example::new(random_dense(400, 128), vec![3]);

    group.bench_function("train_step_128_64_10", |b| {
        b.iter(|| {
            black_box(network.train_step(black_box(&[&example]), 0.01));
        })
    });

    // Multi-threaded training (uses all cores via Rayon when num_threads > 1)
    let num_workers = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .max(1);
    let mut config_parallel = make_bench_config();
    config_parallel.network.num_threads = num_workers;
    config_parallel.hogwild = true;
    let network_parallel = Network::new(config_parallel);
    let hogwild = HogwildNetwork::new(network_parallel);
    let batch: Vec<Example> = (0..64)
        .map(|i| Example::new(random_dense(400 + i, 128), vec![(i % 10) as u32]))
        .collect();
    group.bench_function(
        format!("train_parallel_128_64_10_{}threads", num_workers),
        |b| {
            b.iter(|| {
                black_box(hogwild.train_parallel(black_box(&batch), num_workers, 0.01, 1));
            })
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_hash_throughput,
    bench_lsh_index,
    bench_sparse_vector,
    bench_forward,
    bench_train_step,
);
criterion_main!(benches);
