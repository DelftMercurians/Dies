pub mod skills;

mod goalkeeper;
pub use goalkeeper::Goalkeeper;
pub mod attacker;
pub mod dribble_role;
pub mod dummy_role;
pub mod fetcher_role;
pub mod harasser;
pub mod kicker_role;
pub mod passer;
pub mod receiver;
pub mod test_role;
pub mod waller;

use std::{cell::RefCell, collections::HashMap};

use dies_core::{PlayerData, RoleType, WorldData};

use crate::control::PlayerControlInput;

pub struct RoleCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a WorldData,
    pub skill_map: &'a mut HashMap<String, SkillState>,
    pub invoke_counts: RefCell<HashMap<String, usize>>,
}

impl RoleCtx<'_> {
    pub fn new<'a>(
        player: &'a PlayerData,
        world: &'a WorldData,
        skill_map: &'a mut HashMap<String, SkillState>,
    ) -> RoleCtx<'a> {
        RoleCtx {
            player,
            world,
            skill_map,
            invoke_counts: RefCell::new(HashMap::new()),
        }
    }

    pub fn reset_skills(&mut self) {
        self.skill_map.clear();
    }

    pub(crate) fn get_invoke_count_and_increment(&self, key: &str) -> usize {
        let mut invoke_counts = self.invoke_counts.borrow_mut();
        let invoke_count = invoke_counts.entry(key.to_string()).or_insert(0);
        *invoke_count += 1;
        *invoke_count
    }
}

pub struct SkillCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a WorldData,
}

impl<'a> From<&RoleCtx<'a>> for SkillCtx<'a> {
    fn from(val: &RoleCtx<'a>) -> Self {
        SkillCtx {
            player: val.player,
            world: val.world,
        }
    }
}

pub trait Role<S = ()>: Send {
    /// Updates the role and returns the control input for the player
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput;

    /// Returns the type of the role. Default implementation returns `RoleType::Player`.
    fn role_type(&self) -> RoleType {
        RoleType::Player
    }
}

/// The result of a skill execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillResult {
    Success,
    Failure,
}

/// The state of a skill execution
pub enum SkillState {
    InProgress(Box<dyn Skill>),
    Done(SkillResult),
}

/// The progress of a skill execution
#[derive(Debug)]
pub enum SkillProgress {
    Continue(PlayerControlInput),
    Done(SkillResult),
}

impl SkillProgress {
    /// Creates a new `SkillProgress` with a `Success` result
    pub fn success() -> SkillProgress {
        SkillProgress::Done(SkillResult::Success)
    }

    /// Creates a new `SkillProgress` with a `Failure` result
    pub fn failure() -> SkillProgress {
        SkillProgress::Done(SkillResult::Failure)
    }

    pub fn map_input<F>(self, f: F) -> SkillProgress
    where
        F: FnOnce(PlayerControlInput) -> PlayerControlInput,
    {
        match self {
            SkillProgress::Continue(input) => SkillProgress::Continue(f(input)),
            SkillProgress::Done(result) => SkillProgress::Done(result),
        }
    }
}

pub trait Skill: Send {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress;
}

/// A macro to conveniently invoke a skill from a Role's `update` method. It takes
/// care of creating the skill instance and updating it.
///
/// This is the same as [`skill!`], but it does not call return with the input from the
/// skill.
///
/// # Example
///
/// ```ignore
/// match invoke_skill!(ctx, TestSkill::new("arg")) {
///    SkillProgress::Continue(input) => return input,
///    SkillProgress::Done(result) => {},
/// };
/// ```
#[macro_export]
macro_rules! invoke_skill {
    ($ctx:ident, $key:tt, $skill:expr) => {{
        let debug_key = format!("p{}.active_skill", $ctx.player.id);
        let skill_ctx = Into::<$crate::roles::SkillCtx>::into(&$ctx);
        let skill_state = $ctx.skill_map.entry($key.to_string()).or_insert_with(|| {
            dies_core::debug_string(debug_key.clone(), stringify!($skill));
            $crate::roles::SkillState::InProgress(Box::new($skill))
        });
        match skill_state {
            $crate::roles::SkillState::InProgress(skill) => {
                let progress = skill.update(skill_ctx);
                if let $crate::roles::SkillProgress::Done(result) = &progress {
                    dies_core::debug_string(debug_key, "None");
                    *skill_state = $crate::roles::SkillState::Done(*result);
                }
                progress
            }
            $crate::roles::SkillState::Done(result) => $crate::roles::SkillProgress::Done(*result),
        }
    }};
    ($ctx:ident, $skill:expr) => {{
        let line_key = format!("{}-{}:{}", $ctx.player.id, file!(), line!());
        let invoke_count = $ctx.get_invoke_count_and_increment(&line_key);
        let key = format!("{}-{}", line_key, invoke_count);
        $crate::invoke_skill!($ctx, key, $skill)
    }};
}

/// A macro to conveniently invoke a skill from a Role's `update` method. It takes
/// care of creating the skill instance, calling its `update` method, and returning
/// the result.
///
/// Expands to a call to the skill's `update` method. If the skill returns
/// `SkillProgress::Continue`, the macro will return the input. If the skill
/// returns `SkillProgress::Done`, the macro will continue executing the role.
///
/// This is the same as [`invoke_skill!`], but it calls return with the input from the
/// skill if the skill is not done. This macro should be preferred whenever possible.
///
/// **WARNING**: This macro is intended to be used in a Role's `update` method. It should
/// not be called from closures or other functions that are not the `update` method.
///
/// **WARNING**: When calling the macro from a loop, you must ensure the invocations are
/// in the same order on every `update` call. This is because the macro uses a counter to
/// keep track of the active skills. If this is not the case, a custom key must be provided
/// to the macro to ensure the correct order.
///
/// **Good**:
/// ```ignore
/// fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
///  // Targets are consistent across updates -> good
///  for target in &self.targets {
///   skill!(ctx, GoToPositionSkill::new(target.clone()));
///  }
///  PlayerControlInput::default()
/// }
/// ```
///
/// **Bad**:
/// ```compile_fail
/// fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
///   // Calling from closure -> bad
///   self.targets.iter().for_each(|target| {
///    skill!(ctx, GoToPositionSkill::new(target.clone()));
///   });
///   PlayerControlInput::default()
/// }
/// ```
///
/// **Good**:
/// ```ignore
/// fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
///  let players_closest_to_ball = self.get_players_closest_to_ball();
///  // Custom key to ensure order -> good
///  for player in players_closest_to_ball {
///     skill!(ctx, player.id, GoToPositionSkill::new(player.position));
///  }
///  PlayerControlInput::default()
/// }
/// ```
///
/// # Example
///
/// The macro returns `SkillResult` when the skill is done.
///
/// ```ignore
/// fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
///   if let SkillResult::Failure = skill!(ctx, TestSkill::new("arg")) {
///     log::error!("Skill failed");
///   }
///   PlayerControlInput::new()
/// }
/// ```
#[macro_export]
macro_rules! skill {
    ($ctx:ident, $key:tt, $skill:expr) => {{
        match $crate::invoke_skill!($ctx, $key, $skill) {
            $crate::roles::SkillProgress::Continue(input) => return input,
            $crate::roles::SkillProgress::Done(result) => result,
        }
    }};
    ($ctx:ident, $skill:expr) => {
        match $crate::invoke_skill!($ctx, $skill) {
            $crate::roles::SkillProgress::Continue(input) => return input,
            $crate::roles::SkillProgress::Done(result) => result,
        }
    };
}
