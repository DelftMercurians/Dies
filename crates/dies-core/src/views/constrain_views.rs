use crate::{BallFrame, FieldGeometry};

use super::WorldView;

pub trait WithBall: WorldView {
    fn ball(&self) -> &BallFrame;
}

pub trait WithMaybeField {
    fn maybe_field(&self) -> Option<&FieldGeometry>;
}

pub trait WithField {
    fn field(&self) -> &FieldGeometry;
}

pub struct FrameWithBall<'a, W> {
    ball: &'a BallFrame,
    world: &'a W,
}

impl<'a, W: WorldView> FrameWithBall<'a, W> {
    pub fn try_from(world: &'a W) -> Option<Self> {
        world
            .world_frame()
            .ball
            .as_ref()
            .map(|ball| Self { ball, world })
    }

    pub fn ball(&self) -> &BallFrame {
        self.ball
    }
}

impl<'a, W: WorldView> WorldView for FrameWithBall<'a, W> {
    fn world_frame(&self) -> &crate::WorldFrame {
        self.world.world_frame()
    }
}

impl<'a, W: WorldView> WithBall for FrameWithBall<'a, W> {
    fn ball(&self) -> &BallFrame {
        self.ball
    }
}

pub struct FrameWithField<'a, W> {
    field: &'a FieldGeometry,
    world: &'a W,
}

impl<'a, W: WithMaybeField> FrameWithField<'a, W> {
    pub fn try_from(world: &'a W) -> Option<Self> {
        world.maybe_field().map(|field| Self { field, world })
    }

    pub fn field(&self) -> &FieldGeometry {
        self.field
    }
}

impl<'a, W: WithMaybeField> WithField for FrameWithField<'a, W> {
    fn field(&self) -> &FieldGeometry {
        self.field
    }
}

impl<'a, W: WithMaybeField + WorldView> WorldView for FrameWithField<'a, W> {
    fn world_frame(&self) -> &crate::WorldFrame {
        self.world.world_frame()
    }
}

pub struct FrameWithBallAndField<'a, W> {
    ball: &'a BallFrame,
    field: &'a FieldGeometry,
    world: &'a W,
}

impl<'a, W: WithMaybeField + WorldView> FrameWithBallAndField<'a, W> {
    pub fn try_from(world: &'a W) -> Option<Self> {
        world
            .world_frame()
            .ball
            .as_ref()
            .and_then(|ball| world.maybe_field().map(|field| Self { ball, field, world }))
    }

    pub fn ball(&self) -> &BallFrame {
        self.ball
    }

    pub fn field(&self) -> &FieldGeometry {
        self.field
    }
}

impl<'a, W: WithMaybeField + WorldView> WithBall for FrameWithBallAndField<'a, W> {
    fn ball(&self) -> &BallFrame {
        self.ball
    }
}

impl<'a, W: WithMaybeField + WorldView> WithField for FrameWithBallAndField<'a, W> {
    fn field(&self) -> &FieldGeometry {
        self.field
    }
}

impl<'a, W: WithMaybeField + WorldView> WorldView for FrameWithBallAndField<'a, W> {
    fn world_frame(&self) -> &crate::WorldFrame {
        self.world.world_frame()
    }
}
