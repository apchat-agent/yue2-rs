use anyhow::{ensure, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};
use yue2::{
    pipeline::YuE2Pipeline,
    protocol::{resolve_sampling, GenerationConfig},
    storage, Device,
};

#[derive(Parser)]
#[command(
    version,
    about = "YuE2: style + lyrics → ABC score → 48 kHz stereo song"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a score and audio, retaining the full Python artifact set.
    Generate(Options),
    /// Generate and save an editable ABC plan and exact token arrays.
    Plan(Options),
    /// Render an external or edited ABC score; requires --abc-file.
    Render(Options),
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Auto,
    Cpu,
    Cuda,
    Metal,
}
#[derive(Clone, Copy, ValueEnum)]
enum AudioFormat {
    Flac,
    Wav,
}

#[derive(Args)]
struct Options {
    /// Request JSON, including style, lyrics and optional sampling overrides.
    #[arg(long)]
    request: PathBuf,
    /// Exact destination directory (must be empty).
    #[arg(long)]
    output: PathBuf,
    /// External ABC text; overrides abc and abc_path in the request.
    #[arg(long)]
    abc_file: Option<PathBuf>,
    /// Explicit local YuE2-3B checkpoint; defaults to the offline HF_HOME snapshot.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Explicit local YuE2-Vae checkpoint; defaults to the offline HF_HOME snapshot.
    #[arg(long)]
    vae_dir: Option<PathBuf>,
    /// GenerationConfig JSON, or a saved effective config.json.
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    device: Backend,
    #[arg(long, value_enum, default_value = "flac")]
    audio_format: AudioFormat,
    #[arg(long, alias = "no-progress")]
    quiet: bool,
}

fn device(backend: Backend) -> Result<Device> {
    match backend {
        Backend::Auto if cfg!(feature = "cuda") => device(Backend::Cuda),
        Backend::Auto | Backend::Cpu => Ok(Device::Cpu),
        Backend::Cuda => {
            ensure!(
                std::env::var("CUDA_VISIBLE_DEVICES").map_or(true, |s| s == "0"),
                "Only GPU 0 is authorized; set CUDA_VISIBLE_DEVICES=0"
            );
            // Before any CUDA initialization, including the request-owned RNG device.
            std::env::set_var("CUDA_VISIBLE_DEVICES", "0");
            Ok(Device::new_cuda(0).context("CUDA requires a --features cuda build and GPU 0")?)
        }
        Backend::Metal => {
            Ok(Device::new_metal(0).context("Metal requires a --features metal build")?)
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let start = Instant::now();
    let (options, plan_only, render) = match cli.command {
        Command::Generate(o) => (o, false, false),
        Command::Plan(o) => (o, true, false),
        Command::Render(o) => (o, false, true),
    };
    ensure!(
        !render || options.abc_file.is_some(),
        "render requires --abc-file"
    );
    let (request, abc_sampling, semantic_sampling) =
        storage::read_request(&options.request, options.abc_file.as_deref())?;
    let mut config = if let Some(path) = &options.config {
        let value: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        GenerationConfig::from_dict(value.get("generation").unwrap_or(&value))?
    } else {
        GenerationConfig::default()
    };
    config.abc = resolve_sampling(abc_sampling.as_ref(), &config.abc)?;
    config.semantic = resolve_sampling(semantic_sampling.as_ref(), &config.semantic)?;
    let model_dir = options
        .model_dir
        .map(Ok)
        .unwrap_or_else(|| yue2::model::snapshot_dir("YuE2-3B"))?;
    let vae_dir = options
        .vae_dir
        .map(Ok)
        .unwrap_or_else(|| yue2::model::snapshot_dir("YuE2-Vae"))?;
    // Refuse overwrite before any checkpoint loading or GPU allocation.
    storage::ensure_empty_directory(&options.output)?;
    let execute = || -> Result<Value> {
        // SAFETY: explicit checkpoint directories are read-only for this process;
        // users must keep them immutable, as documented for the library API.
        let mut pipe = unsafe {
            YuE2Pipeline::from_pretrained(model_dir, vae_dir, device(options.device)?, config)?
        };
        pipe.progress = !options.quiet;
        if plan_only {
            let plan = pipe.plan(&request)?;
            plan.save(&options.output)?;
            Ok(
                json!({"stage": "plan", "output": options.output, "truncated": plan.truncated,
                "timing": plan.timing, "wall_seconds": start.elapsed().as_secs_f64()}),
            )
        } else {
            let result = pipe.generate(&request)?;
            let format = match options.audio_format {
                AudioFormat::Flac => "flac",
                AudioFormat::Wav => "wav",
            };
            let receipt = result.save_artifacts_with_format(&options.output, format)?;
            Ok(
                json!({"status": "complete", "output": options.output, "truncated": receipt["truncated"],
                "seconds": result.timing["e2e_seconds"], "audio_seconds": receipt["audio_seconds"],
                "wall_seconds": start.elapsed().as_secs_f64()}),
            )
        }
    };
    match execute() {
        Ok(summary) => {
            println!("{}", serde_json::to_string(&summary)?);
            Ok(())
        }
        Err(error) => {
            storage::write_json(
                options.output.join("failure.json"),
                &json!({
                "status": "failed", "type": "Error", "reason": format!("{error:#}"), "request": request}),
            )?;
            Err(error)
        }
    }
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("yue2: {error:#}");
        std::process::exit(1);
    }
}
