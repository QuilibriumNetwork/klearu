//! Prints a summary table from Criterion benchmark results.
//!
//! Run the llm_inference benchmark first, then run this binary to print a table:
//!
//!   cargo bench --bench llm_inference -p klearu-llm
//!   cargo run --bin bench_table -p klearu-llm [TARGET_CRITERION_DIR]
//!
//! Default directory is `target/criterion` (relative to current working directory).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| {
        if let Ok(t) = env::var("CARGO_TARGET_DIR") {
            format!("{}/criterion", t)
        } else {
            "target/criterion".to_string()
        }
    });
    let root = Path::new(&root);

    let root = if root.is_dir() {
        root.to_path_buf()
    } else if env::args().nth(1).is_some() {
        eprintln!("Not a directory: {}", root.display());
        eprintln!("Run first: cargo bench --bench llm_inference -p klearu-llm");
        std::process::exit(1);
    } else {
        // Try from crate dir when run via cargo run -p klearu-llm
        let fallback = Path::new("../target/criterion");
        if fallback.is_dir() {
            fallback.to_path_buf()
        } else {
            eprintln!("Not a directory: {}", root.display());
            eprintln!("Run first: cargo bench --bench llm_inference -p klearu-llm");
            eprintln!("Or pass the criterion dir: cargo run --bin bench_table -p klearu-llm -- /path/to/target/criterion");
            std::process::exit(1);
        }
    };

    let mut by_group: BTreeMap<String, Vec<(String, f64)>> = BTreeMap::new();

    walk_criterion(&root, &root, &mut by_group);

    if by_group.is_empty() {
        eprintln!("No estimates found under {}", root.display());
        std::process::exit(1);
    }

    print_markdown_table(&by_group);
}

fn walk_criterion(
    root: &Path,
    dir: &Path,
    by_group: &mut BTreeMap<String, Vec<(String, f64)>>,
) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // "new" or "base" contains estimates.json; the parent is the benchmark id
            if name == "new" || name == "base" {
                if let Some(est) = read_estimate(&path) {
                    let rel = path.parent().and_then(|p| p.strip_prefix(root).ok());
                    if let Some(rel) = rel {
                        let parts: Vec<&str> = rel.components().filter_map(|c| c.as_os_str().to_str()).collect();
                        if parts.is_empty() {
                            continue;
                        }
                        let group = parts[0].to_string();
                        let id = if parts.len() > 1 {
                            parts[1..].join(" / ")
                        } else {
                            String::new()
                        };
                        by_group.entry(group).or_default().push((id, est));
                    }
                }
            } else {
                walk_criterion(root, &path, by_group);
            }
        }
    }
}

fn read_estimate(dir: &Path) -> Option<f64> {
    let path = dir.join("estimates.json");
    let json = fs::read_to_string(&path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&json).ok()?;
    let mean = val.get("mean")?;
    let point = mean.get("point_estimate")?.as_f64()?;
    // Criterion uses nanoseconds
    Some(point)
}

fn format_time_ns(ns: f64) -> String {
    if ns >= 1_000_000_000.0 {
        format!("{:.2} s", ns / 1_000_000_000.0)
    } else if ns >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else {
        format!("{:.2} ns", ns)
    }
}

fn print_markdown_table(by_group: &BTreeMap<String, Vec<(String, f64)>>) {
    for (group, rows) in by_group {
        println!("## {}", group);
        println!();
        println!("| Variant | Time |");
        println!("| --- | --- |");
        let mut rows = rows.clone();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for (id, ns) in rows {
            let time = format_time_ns(ns);
            let id_display = if id.is_empty() { "—" } else { id.as_str() };
            println!("| {} | {} |", id_display, time);
        }
        println!();
    }
}
