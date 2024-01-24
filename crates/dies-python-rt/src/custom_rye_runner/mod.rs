mod custom_rye;
mod download;

pub use custom_rye::CustomRyeRunner;
pub use download::{get_custom_rye_dir, get_or_download_python, remove_python};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cache_dir() {
        let cache_dir = get_custom_rye_dir().unwrap();
        println!("cache_dir: {}", cache_dir.display());
        assert!(cache_dir.exists());
    }

    #[test]
    fn test_remove_python() {
        remove_python().unwrap();
    }

    #[test]
    fn test_get_or_download_python() {
        let python_bin = get_or_download_python().unwrap();
        println!("python_bin: {}", python_bin.display());
        assert!(python_bin.exists());
    }

    // #[test]
    // fn test_sync() {
    //     todo!("test sync");
    // }
}

// #[tokio::test]
// async fn test_download_python() {
//     let cache_dir = get_custom_rye_dir().unwrap();
//     let python_zip = cache_dir.join("python-build-standalone.zip");
//     if python_zip.exists() {
//         std::fs::remove_file(&python_zip).unwrap();
//     }
//     download().await;
//     assert!(python_zip.exists());
// }
