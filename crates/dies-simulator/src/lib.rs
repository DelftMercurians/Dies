pub fn add(left: usize, right: usize) -> usize {
    left + right
}

mod run_simulation;
pub use run_simulation::*;

mod consts;
pub use consts::*;

mod box_spawner;
pub use box_spawner::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_simulation() {
        run_simulation::run_simulation();
    }
}
