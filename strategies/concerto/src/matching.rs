//! Minimum-cost assignment (Hungarian / Kuhn–Munkres), O(n³).
//!
//! Trivially cheap for the ≤6 robots × ~9 roles we match each recalc. Costs may
//! be negative (importance is subtracted from redirect time). Requires
//! `rows <= cols`; each row is matched to a distinct column minimizing total cost.

const INF: f64 = 1e18;

/// Solve the rectangular assignment problem for `cost` (rows × cols, rows ≤ cols).
///
/// Returns `row -> Some(col)` for every row. Panics if `rows > cols`.
pub fn assign_min_cost(cost: &[Vec<f64>]) -> Vec<Option<usize>> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    assert!(n <= m, "assign_min_cost requires rows <= cols");

    // 1-indexed potentials and matching (e-maxx formulation).
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut p = vec![0usize; m + 1]; // p[j] = row (1-indexed) matched to col j
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![INF; m + 1];
        let mut used = vec![false; m + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0usize;

            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        // Augment along the alternating path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![None; n];
    for j in 1..=m {
        if p[j] != 0 {
            result[p[j] - 1] = Some(j - 1);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(cost: &[Vec<f64>], assign: &[Option<usize>]) -> f64 {
        assign
            .iter()
            .enumerate()
            .map(|(r, c)| cost[r][c.unwrap()])
            .sum()
    }

    #[test]
    fn square_identity_is_optimal() {
        // Cheapest is the diagonal.
        let cost = vec![
            vec![1.0, 9.0, 9.0],
            vec![9.0, 1.0, 9.0],
            vec![9.0, 9.0, 1.0],
        ];
        let a = assign_min_cost(&cost);
        assert_eq!(a, vec![Some(0), Some(1), Some(2)]);
        assert!((total(&cost, &a) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn forces_swap_for_global_optimum() {
        // Greedy would take (0,0)=1 then be forced into (1,1)=100. Optimal swaps.
        let cost = vec![vec![1.0, 2.0], vec![2.0, 100.0]];
        let a = assign_min_cost(&cost);
        assert_eq!(a, vec![Some(1), Some(0)]);
        assert!((total(&cost, &a) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn rectangular_leaves_expensive_role_unfilled() {
        // 2 robots, 3 roles. Each robot should take its cheap role; role 2 unused.
        let cost = vec![vec![1.0, 50.0, 50.0], vec![50.0, 1.0, 50.0]];
        let a = assign_min_cost(&cost);
        assert_eq!(a[0], Some(0));
        assert_eq!(a[1], Some(1));
    }

    #[test]
    fn handles_negative_costs() {
        let cost = vec![vec![-5.0, -1.0], vec![-1.0, -5.0]];
        let a = assign_min_cost(&cost);
        assert_eq!(a, vec![Some(0), Some(1)]);
        assert!((total(&cost, &a) + 10.0).abs() < 1e-9);
    }
}
