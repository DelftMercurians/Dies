use rocket::fs::{relative, FileServer};

use dies_core::WorldData;
use dies_protos::{
    ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame, SSL_DetectionRobot},
    ssl_vision_geometry::{SSL_GeometryData, SSL_GeometryFieldSize},
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use dies_world::{WorldConfig, WorldTracker};

pub fn run_web_ui() {
    let handler = std::thread::spawn(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(lunch_1())
    });

    handler.join().unwrap();
}

async fn lunch_1() {
    let rocket = rocket::build()
        // .mount("/", routes![index])
        .mount("/", FileServer::from(relative!("static")));
    let _ = rocket.launch().await;

    let world_data = set_word_data();
}

fn set_word_data() -> WorldData {
    let mut tracker = WorldTracker::new(WorldConfig {
        is_blue: true,
        initial_opp_goal_x: 1.0,
    });

    // First detection frame
    let mut frame = SSL_DetectionFrame::new();
    frame.set_t_capture(1.0);
    // Add ball
    let mut ball = SSL_DetectionBall::new();
    ball.set_x(0.0);
    ball.set_y(0.0);
    ball.set_z(0.0);
    frame.balls.push(ball.clone());
    // Add player
    let mut player = SSL_DetectionRobot::new();
    player.set_robot_id(1);
    player.set_x(100.0);
    player.set_y(200.0);
    player.set_orientation(0.0);
    frame.robots_blue.push(player.clone());
    let mut packet_detection = SSL_WrapperPacket::new();
    packet_detection.detection = Some(frame.clone()).into();

    // Add field geometry
    let mut geom = SSL_GeometryData::new();
    let mut field = SSL_GeometryFieldSize::new();
    field.set_field_length(9000);
    field.set_field_width(6000);
    field.set_goal_width(1000);
    field.set_goal_depth(200);
    field.set_boundary_width(300);
    geom.field = Some(field).into();
    let mut packet_geom = SSL_WrapperPacket::new();
    packet_geom.geometry = Some(geom).into();

    tracker.update_from_protobuf(&packet_detection);
    tracker.update_from_protobuf(&packet_detection);

    tracker.get().unwrap()
}
