//! Active-set solution polishing (roadmap 2.3).
//!
//! An ADMM iterate that satisfies the stopping tolerances (~1e-6) rarely
//! satisfies much more. Polishing takes a `Solved` iterate, guesses the
//! active set from it, treats those constraints as equalities, and solves
//! the resulting KKT system directly — through the same SMW reduction the
//! iterations use, so each pass costs one `O(n r'^2 + r'^3)` factorization
//! with `r' = factors + equalities + active inequalities`. Active bounds
//! fold into the base diagonal exactly like the box block in the ADMM
//! x-update, so long-only portfolios with many variables pinned at zero do
//! not grow the reduced dimension.
//!
//! The initial guess marks a constraint active when the iterate sits on it
//! within a scaled multiple of the stopping tolerance and the multiplier
//! does not point the other way. Because an ADMM-accurate iterate can
//! misclassify marginal constraints, the guess is then refined for a few
//! bounded passes in the classic active-set manner: rows whose polished
//! multiplier has the wrong sign are dropped, rows the polished iterate
//! violates are added, and the system is re-solved.
//!
//! Each pass solves the saddle system regularized by
//! `polish_regularization` (so it stays factorable even when the guess is
//! degenerate) and removes the regularization error with
//! `polish_refinement_iterations` rounds of iterative refinement against the
//! *unregularized* KKT matrix (OSQP-style).
//!
//! Polishing is strictly opportunistic: every candidate is re-audited with
//! [`check_kkt`] on the original data and the best candidate is adopted only
//! when its worst KKT residual improves on the ADMM iterate's. A failed
//! factorization or a non-improving candidate leaves the solution untouched,
//! so enabling polish changes residual quality, never correctness.
//! Certificates are never polished — only `Solved` iterates enter this
//! module.

use crate::{
    kkt::{check_kkt, DualVariables, KktResiduals},
    linalg::{covariance_columns, SmwSystem},
    matrix::{dot, Matrix},
    problem::QpProblem,
    solver::SolverSettings,
};

/// Active-set refinement passes. Each pass costs one reduced factorization;
/// well-classified problems stop after the first (the set stops changing).
const MAX_PASSES: usize = 4;

/// Violations larger than this scaled threshold add a row to the next
/// pass's active set; smaller ones are refinement-level noise.
const VIOLATION_THRESHOLD: f64 = 1.0e-12;

/// A polished iterate whose worst KKT residual improved on the ADMM one.
pub(crate) struct Polished {
    pub x: Vec<f64>,
    pub dual: DualVariables,
    pub residuals: KktResiduals,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Lower,
    Upper,
    /// L1 no-trade pin: the variable sits on the kink of the
    /// [`L1Term`](crate::L1Term) and is held at its anchor.
    Anchor,
}

/// Constraint rows treated as equalities by one polishing pass.
///
/// With an L1 term, every variable is additionally classified by its
/// assumed trade direction: `+1` above the anchor (the signed cost joins
/// the linear term), `-1` below, `0` on the kink (pinned at the anchor
/// like an active bound, unless a bound pin already holds it).
#[derive(Clone, PartialEq, Eq)]
struct ActiveSet {
    /// Active inequality row indices, ascending.
    inequalities: Vec<usize>,
    /// Pinned variables — active bounds and L1 no-trade anchors — with the
    /// pinned side, ascending.
    bounds: Vec<(usize, Side)>,
    /// Assumed L1 trade direction per variable; empty when the problem has
    /// no L1 term.
    l1_signs: Vec<i8>,
}

impl ActiveSet {
    fn bound_value(&self, problem: &QpProblem, entry: usize) -> f64 {
        let (index, side) = self.bounds[entry];
        match side {
            Side::Lower => problem.lower_bounds[index],
            Side::Upper => problem.upper_bounds[index],
            Side::Anchor => {
                problem
                    .l1
                    .as_ref()
                    .expect("anchor pins require an L1 term")
                    .anchor[index]
            }
        }
    }
}

/// Attempts an active-set polish of a solved iterate.
///
/// Returns `None` when no candidate improves the worst KKT residual; the
/// caller keeps the ADMM iterate in that case.
pub(crate) fn polish(
    problem: &QpProblem,
    x: &[f64],
    dual: &DualVariables,
    residuals: &KktResiduals,
    settings: &SolverSettings,
) -> Option<Polished> {
    let mut active = initial_guess(problem, x, dual, settings);
    let mut best: Option<Polished> = None;
    let mut best_worst = worst(residuals);

    for _ in 0..MAX_PASSES {
        let Some(candidate) = solve_active_set(problem, &active, settings) else {
            break;
        };
        let candidate_worst = worst(&candidate.residuals);
        let improved_active = refine_active_set(problem, &candidate, &active);
        if candidate_worst < best_worst {
            best_worst = candidate_worst;
            best = Some(candidate);
        }
        match improved_active {
            Some(next) => active = next,
            // The active set is stable: another pass would re-solve the
            // same system.
            None => break,
        }
    }
    best
}

fn worst(residuals: &KktResiduals) -> f64 {
    residuals
        .primal
        .max(residuals.dual)
        .max(residuals.complementarity)
}

/// Active-set guess from the ADMM iterate: a constraint is active when the
/// iterate sits on it within a scaled multiple of the stopping tolerance
/// *and* its multiplier does not point the other way. Tightness alone
/// over-pins near degeneracy; multipliers alone misclassify, because an
/// interior variable can carry harmless sign noise (~1e-19) that would
/// wrongly pin it to a distant bound.
fn initial_guess(
    problem: &QpProblem,
    x: &[f64],
    dual: &DualVariables,
    settings: &SolverSettings,
) -> ActiveSet {
    let tight = |value: f64, target: f64| {
        let scale = 1.0_f64.max(value.abs()).max(target.abs());
        (target - value).abs()
            <= 10.0 * (settings.absolute_tolerance + settings.relative_tolerance * scale)
    };
    let inequality_values = problem.inequalities.matrix.mul_vec(x);
    let inequalities: Vec<usize> = (0..problem.inequalities.len())
        .filter(|&row| tight(inequality_values[row], problem.inequalities.rhs[row]))
        .collect();
    let mut bounds: Vec<(usize, Side)> = (0..x.len())
        .filter_map(|index| {
            let multiplier = dual.bounds[index];
            let lower = problem.lower_bounds[index];
            let upper = problem.upper_bounds[index];
            let at_upper = upper.is_finite() && tight(x[index], upper) && multiplier >= 0.0;
            let at_lower = lower.is_finite() && tight(x[index], lower) && multiplier <= 0.0;
            match (at_lower, at_upper) {
                (false, true) => Some((index, Side::Upper)),
                (true, false) => Some((index, Side::Lower)),
                // A fixed variable (lower == upper within tolerance) picks
                // the side its multiplier points at; the values coincide.
                (true, true) => Some((
                    index,
                    if multiplier > 0.0 {
                        Side::Upper
                    } else {
                        Side::Lower
                    },
                )),
                (false, false) => None,
            }
        })
        .collect();
    let l1_signs = problem.l1.as_ref().map_or_else(Vec::new, |term| {
        (0..x.len())
            .map(|index| {
                let offset = x[index] - term.anchor[index];
                if tight(x[index], term.anchor[index]) {
                    0
                } else if offset > 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    });
    add_anchor_pins(&mut bounds, &l1_signs);
    ActiveSet {
        inequalities,
        bounds,
        l1_signs,
    }
}

/// Pins every no-trade variable (`sign == 0`) at its anchor, unless a bound
/// pin already holds it — a doubly pinned variable would split one
/// multiplier across two rows arbitrarily and fail the sign audit.
fn add_anchor_pins(bounds: &mut Vec<(usize, Side)>, l1_signs: &[i8]) {
    for (index, sign) in l1_signs.iter().enumerate() {
        if *sign == 0 && !bounds.iter().any(|(pinned, _)| *pinned == index) {
            bounds.push((index, Side::Anchor));
        }
    }
    bounds.sort_by_key(|(index, _)| *index);
}

/// Classic active-set update from a polished candidate: drop rows whose
/// multiplier points the wrong way, add rows the candidate violates. For an
/// L1 term, a no-trade pin whose implied multiplier escapes `[-c_i, c_i]`
/// becomes a trade in the direction it escaped, and a trading variable that
/// crossed its anchor is re-pinned. Returns `None` when the set is
/// unchanged.
fn refine_active_set(
    problem: &QpProblem,
    candidate: &Polished,
    active: &ActiveSet,
) -> Option<ActiveSet> {
    let violated = |value: f64, target: f64| {
        let scale = 1.0_f64.max(value.abs()).max(target.abs());
        value - target > VIOLATION_THRESHOLD * scale
    };
    let inequality_values = problem.inequalities.matrix.mul_vec(&candidate.x);
    let inequalities: Vec<usize> = (0..problem.inequalities.len())
        .filter(|&row| {
            if active.inequalities.contains(&row) {
                candidate.dual.inequalities[row] >= 0.0
            } else {
                violated(inequality_values[row], problem.inequalities.rhs[row])
            }
        })
        .collect();
    // Anchor pins are rebuilt from the refreshed trade signs below; only
    // genuine bound pins flow through the keep/add logic.
    let mut bounds: Vec<(usize, Side)> = (0..candidate.x.len())
        .filter_map(|index| {
            let position = active
                .bounds
                .iter()
                .find(|(active, side)| *active == index && *side != Side::Anchor);
            if let Some(&(_, side)) = position {
                let multiplier = candidate.dual.bounds[index];
                let kept = match side {
                    Side::Lower => multiplier <= 0.0,
                    Side::Upper => multiplier >= 0.0,
                    Side::Anchor => unreachable!("anchor pins are filtered out"),
                };
                return kept.then_some((index, side));
            }
            if violated(candidate.x[index], problem.upper_bounds[index]) {
                Some((index, Side::Upper))
            } else if violated(problem.lower_bounds[index], candidate.x[index]) {
                Some((index, Side::Lower))
            } else {
                None
            }
        })
        .collect();
    let l1_signs: Vec<i8> = problem.l1.as_ref().map_or_else(Vec::new, |term| {
        (0..candidate.x.len())
            .map(|index| {
                let previous = active.l1_signs[index];
                if previous == 0 {
                    // Pinned on the kink: unpin toward the side the implied
                    // multiplier escaped to. (Bound-pinned variables carry a
                    // zero L1 multiplier and stay put.)
                    let multiplier = candidate.dual.l1[index];
                    if multiplier > term.costs[index] {
                        1
                    } else if multiplier < -term.costs[index] {
                        -1
                    } else {
                        0
                    }
                } else {
                    // Trading: re-pin when the candidate crossed the anchor.
                    let offset = candidate.x[index] - term.anchor[index];
                    let scale = 1.0_f64
                        .max(candidate.x[index].abs())
                        .max(term.anchor[index].abs());
                    if f64::from(previous) * offset < -VIOLATION_THRESHOLD * scale {
                        0
                    } else {
                        previous
                    }
                }
            })
            .collect()
    });
    add_anchor_pins(&mut bounds, &l1_signs);
    let next = ActiveSet {
        inequalities,
        bounds,
        l1_signs,
    };
    (next != *active).then_some(next)
}

/// Solves the KKT system of the equality-constrained QP defined by one
/// active set and audits the result with [`check_kkt`].
fn solve_active_set(
    problem: &QpProblem,
    active: &ActiveSet,
    settings: &SolverSettings,
) -> Option<Polished> {
    let delta = settings.polish_regularization;
    let n = problem.quadratic.dimension();
    let equality_count = problem.equalities.len();
    let stacked_count = equality_count + active.inequalities.len();

    // Regularized KKT matrix, eliminated onto the primal block:
    // `H = Q + delta*I + (1/delta) A' A` with `A` the active rows. General
    // rows enter as SMW columns scaled by `1/sqrt(delta)`; active-bound rows
    // are unit vectors, so `(1/delta) e_i e_i'` folds into the diagonal.
    let covariance = covariance_columns(&problem.quadratic).ok()?;
    let factor_count = problem.quadratic.factor_count();
    let mut columns = Matrix::zeros(n, factor_count + stacked_count);
    for row in 0..n {
        for col in 0..factor_count {
            columns[(row, col)] = covariance[(row, col)];
        }
    }
    let row_weight = 1.0 / delta.sqrt();
    for constraint in 0..equality_count {
        for variable in 0..n {
            columns[(variable, factor_count + constraint)] =
                row_weight * problem.equalities.matrix[(constraint, variable)];
        }
    }
    for (offset, &row) in active.inequalities.iter().enumerate() {
        for variable in 0..n {
            columns[(variable, factor_count + equality_count + offset)] =
                row_weight * problem.inequalities.matrix[(row, variable)];
        }
    }
    let mut diagonal: Vec<f64> = problem
        .quadratic
        .diagonal
        .iter()
        .map(|value| value + delta)
        .collect();
    for &(index, _) in &active.bounds {
        diagonal[index] += 1.0 / delta;
    }
    let system = SmwSystem::factor(&diagonal, columns).ok()?;

    // Right-hand side of the target (unregularized) KKT system. Trading
    // variables see the signed L1 cost as part of the linear term: on their
    // side of the kink the term is exactly `sign * c_i * (x_i - a_i)`.
    let objective_rhs: Vec<f64> = match &problem.l1 {
        Some(term) => problem
            .linear
            .iter()
            .enumerate()
            .map(|(index, value)| -(value + f64::from(active.l1_signs[index]) * term.costs[index]))
            .collect(),
        None => problem.linear.iter().map(|value| -value).collect(),
    };
    let mut stacked_rhs = problem.equalities.rhs.clone();
    stacked_rhs.extend(
        active
            .inequalities
            .iter()
            .map(|&row| problem.inequalities.rhs[row]),
    );
    let bound_rhs: Vec<f64> = (0..active.bounds.len())
        .map(|entry| active.bound_value(problem, entry))
        .collect();

    let mut candidate = solve_regularized(
        problem,
        &system,
        active,
        delta,
        &objective_rhs,
        &stacked_rhs,
        &bound_rhs,
    );
    for _ in 0..settings.polish_refinement_iterations {
        let applied = apply_kkt(problem, active, &candidate);
        let residual = KktVectors {
            x: subtract(&objective_rhs, &applied.x),
            stacked: subtract(&stacked_rhs, &applied.stacked),
            bounds: subtract(&bound_rhs, &applied.bounds),
        };
        let correction = solve_regularized(
            problem,
            &system,
            active,
            delta,
            &residual.x,
            &residual.stacked,
            &residual.bounds,
        );
        add_assign(&mut candidate.x, &correction.x);
        add_assign(&mut candidate.stacked, &correction.stacked);
        add_assign(&mut candidate.bounds, &correction.bounds);
    }

    // NaN or infinity from a borderline factorization must reject itself;
    // `check_kkt`'s max-folds can drop non-final NaNs, so test explicitly.
    if candidate
        .x
        .iter()
        .chain(&candidate.stacked)
        .chain(&candidate.bounds)
        .any(|value| !value.is_finite())
    {
        return None;
    }

    let mut inequality_dual = vec![0.0; problem.inequalities.len()];
    for (offset, &row) in active.inequalities.iter().enumerate() {
        inequality_dual[row] = candidate.stacked[equality_count + offset];
    }
    let mut bound_dual = vec![0.0; n];
    let mut l1_dual = if problem.l1.is_some() {
        vec![0.0; n]
    } else {
        Vec::new()
    };
    for (&(index, side), multiplier) in active.bounds.iter().zip(&candidate.bounds) {
        if side == Side::Anchor {
            l1_dual[index] = *multiplier;
        } else {
            bound_dual[index] = *multiplier;
        }
    }
    if let Some(term) = &problem.l1 {
        for (index, sign) in active.l1_signs.iter().enumerate() {
            if *sign != 0 {
                l1_dual[index] = f64::from(*sign) * term.costs[index];
            }
        }
    }
    let dual = DualVariables {
        equalities: candidate.stacked[..equality_count].to_vec(),
        inequalities: inequality_dual,
        bounds: bound_dual,
        l1: l1_dual,
    };
    let residuals = check_kkt(problem, &candidate.x, &dual).ok()?;
    Some(Polished {
        x: candidate.x,
        dual,
        residuals,
    })
}

/// Primal block plus multipliers for the stacked (equality + active
/// inequality) rows and the active-bound rows.
struct KktVectors {
    x: Vec<f64>,
    stacked: Vec<f64>,
    bounds: Vec<f64>,
}

/// Solves the regularized saddle system
/// `[Q + delta*I, A'; A, -delta*I] [x; y] = [g_x; g_a]` by eliminating the
/// multiplier block: `H x = g_x + (1/delta) A' g_a`, then
/// `y = (A x - g_a) / delta`.
fn solve_regularized(
    problem: &QpProblem,
    system: &SmwSystem,
    active: &ActiveSet,
    delta: f64,
    objective_rhs: &[f64],
    stacked_rhs: &[f64],
    bound_rhs: &[f64],
) -> KktVectors {
    let equality_count = problem.equalities.len();
    let mut x = objective_rhs.to_vec();
    let scaled_equalities: Vec<f64> = stacked_rhs[..equality_count]
        .iter()
        .map(|value| value / delta)
        .collect();
    problem
        .equalities
        .matrix
        .transpose_mul_add(&scaled_equalities, &mut x);
    for (offset, &row) in active.inequalities.iter().enumerate() {
        let scale = stacked_rhs[equality_count + offset] / delta;
        for (target, coefficient) in x.iter_mut().zip(problem.inequalities.matrix.row(row)) {
            *target += scale * coefficient;
        }
    }
    for (&(index, _), value) in active.bounds.iter().zip(bound_rhs) {
        x[index] += value / delta;
    }
    system.solve_in_place(&mut x);

    let mut stacked = problem.equalities.matrix.mul_vec(&x);
    stacked.extend(
        active
            .inequalities
            .iter()
            .map(|&row| dot(problem.inequalities.matrix.row(row), &x)),
    );
    for (value, rhs) in stacked.iter_mut().zip(stacked_rhs) {
        *value = (*value - rhs) / delta;
    }
    let bounds: Vec<f64> = active
        .bounds
        .iter()
        .zip(bound_rhs)
        .map(|(&(index, _), rhs)| (x[index] - rhs) / delta)
        .collect();
    KktVectors { x, stacked, bounds }
}

/// Applies the unregularized KKT matrix `[Q, A'; A, 0]` for refinement.
fn apply_kkt(problem: &QpProblem, active: &ActiveSet, z: &KktVectors) -> KktVectors {
    let equality_count = problem.equalities.len();
    let mut x = problem.quadratic.apply(&z.x);
    problem
        .equalities
        .matrix
        .transpose_mul_add(&z.stacked[..equality_count], &mut x);
    for (offset, &row) in active.inequalities.iter().enumerate() {
        let scale = z.stacked[equality_count + offset];
        for (target, coefficient) in x.iter_mut().zip(problem.inequalities.matrix.row(row)) {
            *target += scale * coefficient;
        }
    }
    for (&(index, _), multiplier) in active.bounds.iter().zip(&z.bounds) {
        x[index] += multiplier;
    }
    let mut stacked = problem.equalities.matrix.mul_vec(&z.x);
    stacked.extend(
        active
            .inequalities
            .iter()
            .map(|&row| dot(problem.inequalities.matrix.row(row), &z.x)),
    );
    let bounds: Vec<f64> = active.bounds.iter().map(|&(index, _)| z.x[index]).collect();
    KktVectors { x, stacked, bounds }
}

fn subtract(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| a - b).collect()
}

fn add_assign(target: &mut [f64], values: &[f64]) {
    for (target, value) in target.iter_mut().zip(values) {
        *target += value;
    }
}
