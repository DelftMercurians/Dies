

#[get("/")]
pub fn index() -> &'static str {
    "Hello, world!"
}

// #[launch]
// pub fn lunch_web_page() -> _ {
//     rocket::build().mount("/", routes![index])
// }