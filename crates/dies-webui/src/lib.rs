#[macro_use]
extern crate rocket;

mod web_page;
mod run_web_ui;
pub use run_web_ui::run_web_ui;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use web_page::index;

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
