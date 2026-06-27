//! Continuous positioning leaves. These map onto the IPC `GoToPos` skill (or
//! `Dribble` when carrying the ball): called every frame, parameters update
//! smoothly, and the node stays `Running` forever (positioning never "succeeds").

use std::any::Any;
use std::sync::Arc;

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::SkillCommand;

use super::super::argument::Argument;
use super::super::situation::{BehaviorStatus, RobotSituation};
use super::super::BtCallback;
use super::{BehaviorNode, TickResult};

fn go_to(position: Vector2, heading: Option<Angle>, with_ball: bool) -> SkillCommand {
    if with_ball {
        SkillCommand::Dribble {
            target_pos: position,
            target_heading: heading.unwrap_or_else(|| Angle::from_radians(0.0)),
        }
    } else {
        SkillCommand::GoToPos { position, heading }
    }
}

// ── Continuous ──────────────────────────────────────────────────────────────

pub struct ContinuousNode {
    position: Option<Argument<Vector2>>,
    heading: Option<Argument<Angle>>,
    with_ball: bool,
}

impl ContinuousNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        let position = self
            .position
            .as_ref()
            .map(|p| p.resolve(situation))
            .unwrap_or_else(|| situation.position());
        let heading = self.heading.as_ref().map(|h| h.resolve(situation));
        (
            BehaviorStatus::Running,
            Some(go_to(position, heading, self.with_ball)),
        )
    }
}

impl From<ContinuousNode> for BehaviorNode {
    fn from(n: ContinuousNode) -> Self {
        BehaviorNode::Continuous(n)
    }
}

#[derive(Default)]
pub struct ContinuousBuilder {
    position: Option<Argument<Vector2>>,
    heading: Option<Argument<Angle>>,
    with_ball: bool,
}

impl ContinuousBuilder {
    pub fn position(mut self, position: impl Into<Argument<Vector2>>) -> Self {
        self.position = Some(position.into());
        self
    }

    pub fn heading(mut self, heading: impl Into<Argument<Angle>>) -> Self {
        self.heading = Some(heading.into());
        self
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = true;
        self
    }

    /// Accepted for source compatibility with the old engine; the IPC skills have
    /// no equivalent knob, so it is a no-op.
    pub fn aggressiveness(self, _aggressiveness: f64) -> Self {
        self
    }

    /// No-op (see [`aggressiveness`](Self::aggressiveness)).
    pub fn carefullness(self, _carefullness: f64) -> Self {
        self
    }

    pub fn build(self) -> ContinuousNode {
        ContinuousNode {
            position: self.position,
            heading: self.heading,
            with_ball: self.with_ball,
        }
    }
}

pub fn continuous(_description: impl AsRef<str>) -> ContinuousBuilder {
    ContinuousBuilder::default()
}

// ── StatefulContinuous ──────────────────────────────────────────────────────

type AnyState = Box<dyn Any + Send + Sync>;
type StatefulPositionCallback =
    dyn Fn(&RobotSituation, Option<&AnyState>) -> (Vector2, Option<AnyState>) + Send + Sync;
type StatefulHeadingCallback =
    dyn Fn(&RobotSituation, Option<&AnyState>) -> (Angle, Option<AnyState>) + Send + Sync;

pub struct StatefulContinuousNode {
    position_callback: Option<Arc<StatefulPositionCallback>>,
    heading_callback: Option<Arc<StatefulHeadingCallback>>,
    position_state: Option<AnyState>,
    heading_state: Option<AnyState>,
    with_ball: bool,
}

impl StatefulContinuousNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        let position = if let Some(cb) = &self.position_callback {
            let (pos, new_state) = cb(situation, self.position_state.as_ref());
            self.position_state = new_state;
            pos
        } else {
            situation.position()
        };

        let heading = self.heading_callback.as_ref().map(|cb| {
            let (h, new_state) = cb(situation, self.heading_state.as_ref());
            self.heading_state = new_state;
            h
        });

        (
            BehaviorStatus::Running,
            Some(go_to(position, heading, self.with_ball)),
        )
    }
}

impl From<StatefulContinuousNode> for BehaviorNode {
    fn from(n: StatefulContinuousNode) -> Self {
        BehaviorNode::StatefulContinuous(n)
    }
}

#[derive(Default)]
pub struct StatefulContinuousBuilder {
    position_callback: Option<Arc<StatefulPositionCallback>>,
    heading_callback: Option<Arc<StatefulHeadingCallback>>,
    with_ball: bool,
}

impl StatefulContinuousBuilder {
    pub fn with_stateful_position<S: 'static + Send + Sync>(
        mut self,
        callback: impl Fn(&RobotSituation, Option<&S>) -> (Vector2, Option<S>) + Send + Sync + 'static,
    ) -> Self {
        self.position_callback = Some(Arc::new(move |situation, state| {
            let typed = state.and_then(|s| s.downcast_ref::<S>());
            let (pos, new_state) = callback(situation, typed);
            (pos, new_state.map(|s| Box::new(s) as AnyState))
        }));
        self
    }

    pub fn with_stateful_heading<S: 'static + Send + Sync>(
        mut self,
        callback: impl Fn(&RobotSituation, Option<&S>) -> (Angle, Option<S>) + Send + Sync + 'static,
    ) -> Self {
        self.heading_callback = Some(Arc::new(move |situation, state| {
            let typed = state.and_then(|s| s.downcast_ref::<S>());
            let (h, new_state) = callback(situation, typed);
            (h, new_state.map(|s| Box::new(s) as AnyState))
        }));
        self
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = true;
        self
    }

    pub fn build(self) -> StatefulContinuousNode {
        StatefulContinuousNode {
            position_callback: self.position_callback,
            heading_callback: self.heading_callback,
            position_state: None,
            heading_state: None,
            with_ball: self.with_ball,
        }
    }
}

pub fn stateful_continuous(_description: impl AsRef<str>) -> StatefulContinuousBuilder {
    StatefulContinuousBuilder::default()
}

/// A no-op callback type marker for headings that just want to ignore state.
pub type Ignore = ();

#[allow(dead_code)]
fn _assert_callback_object_safe(_: &dyn BtCallback<Vector2>) {}
