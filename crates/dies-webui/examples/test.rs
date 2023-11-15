#[macro_use] extern crate rocket;

#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(lunch_1());
}

async fn lunch_1() {
    let rocket = rocket::build().mount("/", routes![index]);
    let _ = rocket.launch().await;

    // todo get word data
}

