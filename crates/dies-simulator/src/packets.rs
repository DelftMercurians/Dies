//! SSL packet encoding and scheduled simulator output buffers.
use crate::{
    config::SimulationConfig,
    physics::PhysicsWorld,
    referee::{card_team_info, AutoReferee, SimulationGameState, SIM_GOALKEEPER_ID},
    robot::RobotFleet,
    time::IntervalTrigger,
};
use dies_core::{FieldGeometry, PlayerFeedbackMsg, PlayerId, SideAssignment, TeamColor};
use dies_protos::{
    ssl_gc_referee_message::{
        referee::{self, TeamInfo},
        Referee,
    },
    ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame, SSL_DetectionRobot},
    ssl_vision_geometry::{
        SSL_FieldCircularArc, SSL_FieldLineSegment, SSL_GeometryData, SSL_GeometryFieldSize,
    },
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use std::collections::HashMap;

pub(crate) struct SimOutput {
    pub(crate) detection_interval: IntervalTrigger,
    pub(crate) last_detection_packet: Option<SSL_WrapperPacket>,
    pub(crate) geometry_interval: IntervalTrigger,
    pub(crate) geometry_packet: SSL_WrapperPacket,
    pub(crate) feedback_interval: IntervalTrigger,
    pub(crate) feedback_queue: HashMap<(TeamColor, PlayerId), PlayerFeedbackMsg>,
}

impl SimOutput {
    pub(crate) fn new(config: &SimulationConfig) -> Self {
        let vision_update_step = config.vision_update_step;
        let geometry_interval = config.geometry_interval;
        let feedback_interval = config.feedback_interval;
        let geometry_packet = geometry(&config.field_geometry);
        Self {
            detection_interval: IntervalTrigger::new(vision_update_step),
            last_detection_packet: None,
            geometry_interval: IntervalTrigger::new(geometry_interval),
            geometry_packet: geometry_packet,
            feedback_interval: IntervalTrigger::new(feedback_interval),
            feedback_queue: HashMap::new(),
        }
    }

    pub(crate) fn detection(&mut self) -> Option<SSL_WrapperPacket> {
        self.last_detection_packet.take()
    }

    pub(crate) fn feedback(&mut self) -> HashMap<(TeamColor, PlayerId), PlayerFeedbackMsg> {
        std::mem::take(&mut self.feedback_queue)
    }

    pub(crate) fn geometry(&mut self, now: f64) -> Option<SSL_WrapperPacket> {
        if self.geometry_interval.trigger(now) {
            Some(self.geometry_packet.clone())
        } else {
            None
        }
    }

    pub(crate) fn new_detection_packet(
        &mut self,
        physics: &PhysicsWorld,
        robots: &RobotFleet,
        now: f64,
    ) {
        let mut detection = SSL_DetectionFrame::new();
        detection.set_frame_number(1);
        detection.set_t_capture(now);
        detection.set_t_sent(now);
        detection.set_camera_id(1);

        // Players
        for player in robots.players.iter() {
            let rigid_body = physics
                .rigid_body_set
                .get(player.rigid_body_handle)
                .unwrap();
            let position = rigid_body.position().translation.vector;
            // Add high frequency noise to the position
            // let position = position
            //     + Vector::new(
            //         (rand::random::<f64>() - 0.5) * 5.0,
            //         (rand::random::<f64>() - 0.5) * 5.0,
            //         0.0,
            //     );

            let yaw = rigid_body.rotation().euler_angles().2;
            // Add high frequency noise to the yaw
            // let yaw = yaw + ((rand::random::<f64>() - 0.5) * 4f64).to_radians();

            let mut robot = SSL_DetectionRobot::new();
            robot.set_robot_id(player.id.as_u32());
            robot.set_x(position.x as f32);
            robot.set_y(position.y as f32);
            robot.set_pixel_x(0.0);
            robot.set_pixel_y(0.0);
            robot.set_orientation(yaw as f32);
            robot.set_confidence(1.0);
            if player.team_color == TeamColor::Blue {
                detection.robots_blue.push(robot);
            } else {
                detection.robots_yellow.push(robot);
            }
        }

        // Ball
        if let Some(ball) = physics.ball.as_ref() {
            let mut ball_det = SSL_DetectionBall::new();
            let ball_body = physics.rigid_body_set.get(ball._rigid_body_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;
            ball_det.set_x(ball_position.x as f32);
            ball_det.set_y(ball_position.y as f32);
            ball_det.set_z(ball_position.z as f32);
            ball_det.set_confidence(1.0);
            ball_det.set_pixel_x(0.0);
            ball_det.set_pixel_y(0.0);
            detection.balls.push(ball_det);
        }

        let mut packet = SSL_WrapperPacket::new();
        packet.detection = Some(detection).into();

        self.last_detection_packet = Some(packet);
    }
}

fn geometry(config: &FieldGeometry) -> SSL_WrapperPacket {
    let mut geometry = SSL_GeometryData::new();
    let mut field = SSL_GeometryFieldSize::new();
    field.set_field_length(config.field_length as i32);
    field.set_field_width(config.field_width as i32);
    field.set_goal_width(config.goal_width as i32);
    field.set_goal_depth(config.goal_depth as i32);
    field.set_boundary_width(config.boundary_width as i32);
    field.set_penalty_area_depth(config.penalty_area_depth as i32);
    field.set_penalty_area_width(config.penalty_area_width as i32);
    field.set_center_circle_radius(config.center_circle_radius as i32);
    field.set_goal_center_to_penalty_mark(config.goal_line_to_penalty_mark as i32);
    field.set_ball_radius(config.ball_radius as f32);

    for line in &config.line_segments {
        let mut ssl_segment = SSL_FieldLineSegment::new();
        ssl_segment.set_name(line.name.clone());
        let mut p1 = dies_protos::ssl_vision_geometry::Vector2f::new();
        p1.set_x(line.p1.x as f32);
        p1.set_y(line.p1.y as f32);
        ssl_segment.p1 = Some(p1).into();
        let mut p2 = dies_protos::ssl_vision_geometry::Vector2f::new();
        p2.set_x(line.p2.x as f32);
        p2.set_y(line.p2.y as f32);
        ssl_segment.p2 = Some(p2).into();
        ssl_segment.set_thickness(line.thickness as f32);
        field.field_lines.push(ssl_segment);
    }

    for arc in &config.circular_arcs {
        let mut ssl_arc = SSL_FieldCircularArc::new();
        ssl_arc.set_name(arc.name.clone());
        let mut center = dies_protos::ssl_vision_geometry::Vector2f::new();
        center.set_x(arc.center.x as f32);
        center.set_y(arc.center.y as f32);
        ssl_arc.center = Some(center).into();
        ssl_arc.set_radius(arc.radius as f32);
        ssl_arc.set_a1(arc.a1 as f32);
        ssl_arc.set_a2(arc.a2 as f32);
        ssl_arc.set_thickness(arc.thickness as f32);
        field.field_arcs.push(ssl_arc);
    }

    geometry.field = Some(field).into();
    let mut packet = SSL_WrapperPacket::new();
    packet.geometry = Some(geometry).into();
    packet
}

pub(crate) fn referee_packet(
    state: &AutoReferee,
    now: f64,
    command: referee::Command,
    next_command: Option<referee::Command>,
    action_time_s: Option<f64>,
) -> Referee {
    let mut msg = Referee::new();
    msg.set_command(command);
    msg.packet_timestamp = Some(0);
    msg.set_stage(referee::Stage::NORMAL_FIRST_HALF);
    // Increment so consumers can detect distinct commands (the real GC does).
    msg.command_counter = Some(state.command_counter);
    msg.command_timestamp = Some(0);
    if let Some(next) = next_command {
        msg.set_next_command(next);
    }
    if let Some(t) = action_time_s {
        msg.current_action_time_remaining = Some((t * 1_000_000.0) as i64);
    }
    if let SimulationGameState::BallPlacement { position, .. } = &state.game_state {
        let mut point = referee::Point::new();
        point.set_x(position.x as f32);
        point.set_y(position.y as f32);
        msg.designated_position = Some(point).into();
    }
    let now = now;
    let (blue_max, blue_cards) = card_team_info(&state.blue_card_expiries, now);
    let (yellow_max, yellow_cards) = card_team_info(&state.yellow_card_expiries, now);
    let mut blue_team_info = TeamInfo::new();
    blue_team_info.set_max_allowed_bots(blue_max);
    blue_team_info.set_goalkeeper(SIM_GOALKEEPER_ID);
    blue_team_info.set_score(state.blue_score);
    blue_team_info.yellow_card_times = blue_cards;
    msg.blue = Some(blue_team_info).into();
    let mut yellow_team_info = TeamInfo::new();
    yellow_team_info.set_max_allowed_bots(yellow_max);
    yellow_team_info.set_goalkeeper(SIM_GOALKEEPER_ID);
    yellow_team_info.set_score(state.yellow_score);
    yellow_team_info.yellow_card_times = yellow_cards;
    msg.yellow = Some(yellow_team_info).into();
    // Report the sim's true side so the executor/strategies and the sim's own
    // goal/penalty checks all agree on which team defends which half.
    msg.blue_team_on_positive_half = Some(state.side_assignment == SideAssignment::BlueOnPositive);
    msg
}
