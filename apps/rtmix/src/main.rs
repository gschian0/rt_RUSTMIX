fn main() {
    if let Err(err) = rtmix_gui::run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
