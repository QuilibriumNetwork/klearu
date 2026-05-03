// Link MPSGraph framework. The metal-rs crate already pulls in Metal +
// MetalPerformanceShaders, but MPSGraph lives in its own framework.
fn main() {
    if std::env::var("CARGO_FEATURE_METAL").is_ok() {
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}
