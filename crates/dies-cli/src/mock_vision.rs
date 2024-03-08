use std::time::{Duration, SystemTime};

use dies_core::FieldGeometry;
use dies_protos::{
    ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame, SSL_DetectionRobot},
    ssl_vision_geometry::{
        SSL_FieldCircularArc, SSL_FieldLineSegment, SSL_GeometryData, SSL_GeometryFieldSize,
    },
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use dies_ssl_client::VisionClientConfig;
use nalgebra::{Vector2, Vector3};
use tokio::sync::mpsc;

pub struct MockPlayer {
    id: u8,
    position: Vector2<f32>,
    orientation: f32,
}

pub struct MockBall {
    position: Vector3<f32>,
}

/// A mock vision system for testing.
pub struct MockVision {
    geometry: FieldGeometry,
    own_players: Vec<MockPlayer>,
    opp_players: Vec<MockPlayer>,
    ball: MockBall,
    own_team_blue: bool,
}

impl MockVision {
    pub fn spawn() -> VisionClientConfig {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut detection_timer = tokio::time::interval(Duration::from_millis(30));
            let mut geometry_timer = tokio::time::interval(Duration::from_secs(3));
            let vision = MockVision::new();
            loop {
                tokio::select! {
                    _ = detection_timer.tick() => {
                        let detection = vision.detection();
                        tx.send(detection).unwrap();
                    }
                    _ = geometry_timer.tick() => {
                        let geometry = vision.geometry();
                        tx.send(geometry).unwrap();
                    }
                }
            }
        });
        VisionClientConfig::InMemory(rx)
    }

    pub fn new() -> Self {
        let geometry = FieldGeometry {
            field_length: 11000,
            field_width: 9000,
            goal_width: 1000,
            goal_depth: 200,
            boundary_width: 200,
            line_segments: Vec::new(),
            circular_arcs: Vec::new(),
        };

        let mut own_players = Vec::new();
        for i in 0..6 {
            let x = 0.0;
            let y = (i as f32 - 2.5) * 1000.0;
            let player = MockPlayer {
                id: i,
                position: Vector2::new(x, y),
                orientation: 0.0,
            };
            own_players.push(player);
        }

        let mut opp_players = Vec::new();
        for i in 0..6 {
            let x = 10000.0;
            let y = (i as f32 - 2.5) * 1000.0;
            let player = MockPlayer {
                id: i,
                position: Vector2::new(x, y),
                orientation: 0.0,
            };
            opp_players.push(player);
        }

        let ball = MockBall {
            position: Vector3::new(0.0, 0.0, 0.0),
        };

        Self {
            geometry,
            own_players,
            opp_players,
            ball,
            own_team_blue: true,
        }
    }

    pub fn detection(&self) -> SSL_WrapperPacket {
        let mut detection = SSL_DetectionFrame::new();
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        detection.set_t_capture(t);
        detection.set_t_sent(t);

        self.own_players.iter().for_each(|player| {
            let mut robot = SSL_DetectionRobot::new();
            robot.set_robot_id(player.id as u32);
            robot.set_x(player.position.x);
            robot.set_y(player.position.y);
            robot.set_orientation(player.orientation);
            robot.set_confidence(1.0);
            if self.own_team_blue {
                detection.robots_blue.push(robot);
            } else {
                detection.robots_yellow.push(robot);
            }
        });

        self.opp_players.iter().for_each(|player| {
            let mut robot = SSL_DetectionRobot::new();
            robot.set_robot_id(player.id as u32);
            robot.set_x(player.position.x);
            robot.set_y(player.position.y);
            robot.set_orientation(player.orientation);
            robot.set_confidence(1.0);
            if self.own_team_blue {
                detection.robots_yellow.push(robot);
            } else {
                detection.robots_blue.push(robot);
            }
        });

        let mut ball = SSL_DetectionBall::new();
        ball.set_x(self.ball.position.x);
        ball.set_y(self.ball.position.y);
        ball.set_z(self.ball.position.z);
        ball.set_confidence(1.0);
        detection.balls.push(ball);

        let mut packet = SSL_WrapperPacket::new();
        packet.detection = Some(detection).into();
        packet
    }

    pub fn geometry(&self) -> SSL_WrapperPacket {
        let mut geometry = SSL_GeometryData::new();
        let mut field = SSL_GeometryFieldSize::new();
        field.set_field_length(self.geometry.field_length);
        field.set_field_width(self.geometry.field_width);
        field.set_goal_width(self.geometry.goal_width);
        field.set_goal_depth(self.geometry.goal_depth);
        field.set_boundary_width(self.geometry.boundary_width);

        for line in &self.geometry.line_segments {
            let mut ssl_segment = SSL_FieldLineSegment::new();
            ssl_segment.set_name(line.name.clone());
            let mut p1 = dies_protos::ssl_vision_geometry::Vector2f::new();
            p1.set_x(line.p1.x);
            p1.set_y(line.p1.y);
            ssl_segment.p1 = Some(p1).into();
            let mut p2 = dies_protos::ssl_vision_geometry::Vector2f::new();
            p2.set_x(line.p2.x);
            p2.set_y(line.p2.y);
            ssl_segment.p2 = Some(p2).into();
            ssl_segment.set_thickness(line.thickness);
            field.field_lines.push(ssl_segment);
        }

        for arc in &self.geometry.circular_arcs {
            let mut ssl_arc = SSL_FieldCircularArc::new();
            ssl_arc.set_name(arc.name.clone());
            let mut center = dies_protos::ssl_vision_geometry::Vector2f::new();
            center.set_x(arc.center.x);
            center.set_y(arc.center.y);
            ssl_arc.center = Some(center).into();
            ssl_arc.set_radius(arc.radius);
            ssl_arc.set_a1(arc.a1);
            ssl_arc.set_a2(arc.a2);
            ssl_arc.set_thickness(arc.thickness);
            field.field_arcs.push(ssl_arc);
        }

        geometry.field = Some(field).into();
        let mut packet = SSL_WrapperPacket::new();
        packet.geometry = Some(geometry).into();
        packet
    }
}
