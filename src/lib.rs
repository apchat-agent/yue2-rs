//! YuE2 inference. The CPU backend is the default; CUDA is opt-in.
pub mod model;
pub mod pipeline;
pub mod protocol;
pub mod sampling;
pub mod tokenizer;

pub use candle_core::{DType, Device, Tensor};
