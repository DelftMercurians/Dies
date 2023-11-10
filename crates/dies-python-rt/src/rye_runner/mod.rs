mod download;
mod rye;

pub use rye::{RyeOutput, RyeRunner};

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use super::*;

    #[test]
    fn test_sync() {
        // Create a temporary directory with a pyproject.toml file and a py directory
        let temp_dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp_dir.path().join("py")).unwrap();
        let pyproject_toml = temp_dir.path().join("pyproject.toml");
        std::fs::write(&pyproject_toml, indoc! {"
            [project]
            name = \"test-workspace\"
            version = \"0.1.0\"
            dependencies = []

            [tool.rye]
            managed = true

            [tool.rye.workspace]
        "}).unwrap();

        // Set `DIES_RYE_DIR` to temporary dir
        std::env::set_var("DIES_RYE_DIR", temp_dir.path().join(".rye"));

        // Create a RyeRunner
        let mut runner = RyeRunner::new(&temp_dir).unwrap();
        runner.output(RyeOutput::Stdout);

        println!("Running rye sync");
        runner.sync().unwrap();
        println!("Rye sync done");

        // Ensure that the .venv directory was created
        assert!(temp_dir.path().join(".venv").is_dir());
    }
}