mod skills;

pub mod test_role;

use std::collections::HashMap;

use crate::control::PlayerControlInput;
use dies_core::{PlayerData, WorldData};

pub struct RoleCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a WorldData,
    skill_map: &'a mut HashMap<String, Box<dyn Skill>>,
}

impl RoleCtx<'_> {
    pub fn new<'a>(
        player: &'a PlayerData,
        world: &'a WorldData,
        skill_map: &'a mut HashMap<String, Box<dyn Skill>>,
    ) -> RoleCtx<'a> {
        RoleCtx {
            player,
            world,
            skill_map,
        }
    }
}

impl<'a> Into<SkillCtx<'a>> for &RoleCtx<'a> {
    fn into(self) -> SkillCtx<'a> {
        SkillCtx {
            player: self.player,
            world: self.world,
        }
    }
}

pub trait Role: Send {
    /// Updates the role and returns the control input for the player
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput;
}

pub struct SkillCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a WorldData,
}

pub enum SkillResult {
    Continue(PlayerControlInput),
    Done,
}

pub trait Skill: Send {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillResult;
}

/// A macro to conveniently invoke a skill from a Role's `update` method. It takes
/// care of creating the skill instance, updating it, and cleaning up the skill
/// instance if it is done.
///
/// This is the same as [`skill!`], but it does not call return with the input from the
/// skill.
///
/// # Example
///
/// ```no_run
/// match invoke_skill!(ctx, TestSkill::new("arg")) {
///    SkillResult::Continue(input) => return input,
///    SkillResult::Done => {},
/// };
/// ```
#[macro_export]
macro_rules! invoke_skill {
    ($ctx:ident, $skill:expr) => {{
        let key = format!("{}:{}", file!(), line!());
        let debug_key = format!("p{}.active_skill", $ctx.player.id);
        let skill_ctx = Into::<crate::roles::SkillCtx>::into(&$ctx);
        let skill = $ctx.skill_map.entry(key.clone()).or_insert_with(|| {
            dies_core::debug_string(debug_key.clone(), stringify!($skill));
            Box::new($skill)
        });
        let result = skill.update(skill_ctx);
        if let crate::roles::SkillResult::Done = &result {
            $ctx.skill_map.remove(&key);
            dies_core::debug_string(debug_key, "");
        }
        result
    }};
}

/// A macro to conveniently invoke a skill from a Role's `update` method. It takes
/// care of creating the skill instance, updating it, and cleaning up the skill
/// instance if it is done.
///
/// Expands to a call to the skill's `update` method. If the skill returns
/// `SkillResult::Continue`, the macro will return the input. If the skill
/// returns `SkillResult::Done`, the macro will continue executing the role.
///
/// This is the same as [`invoke_skill!`], but it calls return with the input from the
/// skill if the skill is not done.
///
/// # Example
///
/// ```no_run
/// fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
///   skill!(ctx, TestSkill::new("arg"));
///   PlayerControlInput::new()
/// }
/// ```
#[macro_export]
macro_rules! skill {
    ($ctx:ident, $skill:expr) => {
        match crate::invoke_skill!($ctx, $skill) {
            crate::roles::SkillResult::Continue(input) => return input,
            crate::roles::SkillResult::Done => {}
        };
    };
}