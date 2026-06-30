//! Bounce the ball between two robots with the joint pass coordinator — on a
//! loop. Exercises `ctx.pass` and the typed `PassResult`.
//!
//! Seed a snapshot with robots 0 and 1 and the ball near robot 0 before running.

use scenarios::prelude::*;

fn main() {
    run_scenario(|| {
        let a = PlayerId::new(0);
        let b = PlayerId::new(1);
        Scenario::looping(move || {
            vec![
                Step::pass(a, b, None).timeout(20.0),
                Step::wait(0.5),
                Step::pass(b, a, None).timeout(20.0),
                Step::wait(0.5),
            ]
        })
    });
}
