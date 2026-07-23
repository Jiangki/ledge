//! Wire representations for serde (roadmap 3.3).
//!
//! Ledge serialization exists for **bug reproduction**: dump a problem (and
//! optionally settings, warm start, and the returned solution) on one
//! machine, attach it to an issue, replay it on another. Every type keeps
//! plain derived serde impls except where an invariant or a format
//! limitation forces a custom wire shape:
//!
//! - [`Matrix`](crate::Matrix) travels as `{ rows, cols, data }` and is
//!   rebuilt through `Matrix::new`, so a corrupted dump cannot construct a
//!   matrix whose shape disagrees with its storage.
//! - [`PortfolioProblem`](crate::PortfolioProblem) travels as its builder
//!   inputs and is rebuilt through the same builder methods, so
//!   deserialization enforces exactly the validation construction does.
//! - Variable bounds may legally be infinite, but JSON has no
//!   representation for non-finite floats (`serde_json` writes `null` and
//!   refuses to read it back). Bound vectors therefore travel as
//!   `Option<f64>` per entry, where `None` means "unbounded on this side"
//!   (`-inf` in a lower bound, `+inf` in an upper bound) — see
//!   [`lower_bounds`] / [`upper_bounds`]. The pathological wrong-sign
//!   infinity (a lower bound of `+inf`) also maps to `None`; such problems
//!   are rejected by validation anyway.
//!
//! Solutions from a [`NumericalFailure`](crate::SolveStatus) contain
//! non-finite iterates by definition; those round-trip through
//! self-describing binary formats (for example `postcard`) but not through
//! JSON. Problems and settings never contain non-finite data outside the
//! bounds handled above.

/// (De)serializes a bound vector as `Option<f64>` entries where `None`
/// stands for the direction's natural infinity.
macro_rules! bound_representation {
    ($module:ident, $infinity:expr, $doc:literal) => {
        #[doc = $doc]
        pub(crate) mod $module {
            use serde::{Deserialize, Deserializer, Serialize, Serializer};

            pub(crate) fn serialize<S: Serializer>(
                values: &[f64],
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                let wire: Vec<Option<f64>> = values
                    .iter()
                    .map(|value| value.is_finite().then_some(*value))
                    .collect();
                wire.serialize(serializer)
            }

            pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Vec<f64>, D::Error> {
                let wire = Vec::<Option<f64>>::deserialize(deserializer)?;
                Ok(wire
                    .into_iter()
                    .map(|value| value.unwrap_or($infinity))
                    .collect())
            }
        }
    };
}

bound_representation!(
    lower_bounds,
    f64::NEG_INFINITY,
    "Lower-bound wire format: `None` means unbounded below (`-inf`)."
);
bound_representation!(
    upper_bounds,
    f64::INFINITY,
    "Upper-bound wire format: `None` means unbounded above (`+inf`)."
);
