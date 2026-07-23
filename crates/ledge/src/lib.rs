//! Public Rust API for the Ledge portfolio QP solver.
//!
//! Numerical implementation details live in `ledge-core`; this crate is the
//! import surface intended for Rust applications and the Python bindings.

#![forbid(unsafe_code)]

pub use ledge_core::*;
