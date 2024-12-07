pub const CONFIG_FILE: &str = "config_file";

fn get_config_file() -> std::path::PathBuf {
    let dir = dirs::cache_dir().unwrap();
    dir.join(CONFIG_FILE)
}

pub fn set_server_name(server_name: String) {
    std::fs::write(get_config_file(), server_name).unwrap();
}

pub fn get_server_name() -> String {
    std::fs::read_to_string(get_config_file()).unwrap()
}