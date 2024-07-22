use dies_core::{GameState, PlayerId, WorldData};

use super::{free_kick::FreeAttacker, Strategy, StrategyCtx};
use crate::roles::{
    attacker::{Attacker, AttackerSection, AttackerState},
    harasser::Harasser,
    waller::Waller,
    Goalkeeper, Role,
};

pub struct Attack {
    attackers: Vec<(PlayerId, Attacker)>,
}

impl Default for Attack {
    fn default() -> Self {
        Self::new()
    }
}

impl Attack {
    pub fn new() -> Self {
        Self {
            attackers: Vec::new(),
        }
    }

    pub fn add_attacker(&mut self, player_id: PlayerId) {
        let sections = self
            .attackers
            .iter()
            .map(|(_, attacker)| attacker.section())
            .collect::<Vec<_>>();
        let next_section = if !sections.contains(&AttackerSection::Left) {
            AttackerSection::Left
        } else if !sections.contains(&AttackerSection::Right) {
            AttackerSection::Right
        } else if !sections.contains(&AttackerSection::Mid) {
            AttackerSection::Mid
        } else {
            AttackerSection::Left
        };
        self.attackers.push((
            player_id,
            Attacker::new(AttackerState::Positioning, next_section),
        ));
    }

    fn update(&mut self, ctx: StrategyCtx) {
        let receiver = self
            .attackers
            .iter_mut()
            .find_map(|(_, attacker)| attacker.passed_to_receiver());
        if let Some(receiver) = receiver {
            if let Some((_, receiver)) = self.attackers.iter_mut().find(|(id, _)| *id == receiver) {
                // receiver.receive();
            } else {
                log::error!("Receiver not found: {:?}", receiver);
            }
        }
    }

    fn fetch_ball(&mut self, ctx: &StrategyCtx) {
        if self.we_fetching_ball() {
            for attacker in self.attackers.iter_mut() {
                if !attacker.1.fetching_ball() {
                    attacker.1.start_positioning();
                }
            }
        }

        // find closest player to the ball
        if let Some(ball) = ctx.world.ball.as_ref() {
            let attacker_ids = self.attackers.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let ball_pos = ball.position.xy();
            let closest_player = ctx
                .world
                .own_players
                .iter()
                .filter(|player| attacker_ids.contains(&player.id))
                .min_by_key(|player| {
                    let player_pos = player.position;
                    let diff = player_pos - ball_pos;
                    diff.norm_squared() as i64
                });
            if let Some(active_attacker) = closest_player {
                if let Some((_, attacker)) = self
                    .attackers
                    .iter_mut()
                    .find(|(id, _)| *id == active_attacker.id)
                {
                    attacker.receive();
                }
            }
        }
    }

    fn we_have_ball(&self) -> bool {
        self.attackers
            .iter()
            .any(|(_, attacker)| attacker.has_ball())
    }

    fn we_fetching_ball(&self) -> bool {
        self.attackers
            .iter()
            .any(|(_, attacker)| attacker.fetching_ball())
    }

    fn get_role(&mut self, player_id: PlayerId, ctx: StrategyCtx) -> Option<&mut dyn Role> {
        self.attackers
            .iter_mut()
            .find_map(|(id, role)| if *id == player_id { Some(role) } else { None })
            .map(|role| role as &mut dyn Role)
    }
}

pub struct Defense {
    keeper_id: PlayerId,
    keeper: Goalkeeper,
    wallers: Vec<(PlayerId, Waller)>,
    harasser: Option<(PlayerId, Harasser)>,
}

impl Defense {
    pub fn new(keeper_id: PlayerId) -> Self {
        Self {
            keeper_id,
            keeper: Goalkeeper::new(),
            wallers: Vec::new(),
            harasser: None,
        }
    }

    pub fn add_wallers(&mut self, player_ids: Vec<PlayerId>) {
        for (idx, id) in player_ids.iter().enumerate() {
            self.wallers.push((*id, Waller::new_with_index(idx)));
        }
    }

    pub fn add_harasser(&mut self, player_id: PlayerId) {
        self.harasser = Some((player_id, Harasser::new(120.0)));
    }

    pub fn update(&mut self, _ctx: StrategyCtx) {
        self.keeper
            .set_defenders(self.wallers.iter().map(|w| w.0).collect());
    }
}

pub struct PlayStrategy {
    pub attack: Attack,
    pub defense: Defense,
    last_game_state: dies_core::GameState,
    free_kick_attacker: Option<(PlayerId, FreeAttacker)>,
}

impl PlayStrategy {
    pub fn new(keeper_id: PlayerId) -> Self {
        Self {
            attack: Attack::new(),
            defense: Defense::new(keeper_id),
            last_game_state: dies_core::GameState::Halt,
            free_kick_attacker: None,
        }
    }

    fn get_player_for_kick(&self, ctx: StrategyCtx) -> PlayerId {
        let wallers = self
            .defense
            .wallers
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let harasser = self.defense.harasser.as_ref().map(|(id, _)| *id);
        let attackers = self
            .attack
            .attackers
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let keeper = self.defense.keeper_id;
        if let Some(ball) = ctx.world.ball.as_ref() {
            if ball.position.x < 0.0 {
                // Try harrassers first
                if let Some(harasser) = harasser {
                    return harasser;
                }

                // Try attackers
                let closest_attacker = self.attack.attackers.iter().min_by_key(|(id, _)| {
                    ctx.world
                        .get_player(*id)
                        .map(|p| (p.position - ball.position.xy()).norm())
                        .unwrap_or(f64::MAX) as i64
                });
                if let Some((id, _)) = closest_attacker {
                    return *id;
                }

                // Try wallers
                let closest_waller = self.defense.wallers.iter().min_by_key(|(id, _)| {
                    ctx.world
                        .get_player(*id)
                        .map(|p| (p.position - ball.position.xy()).norm())
                        .unwrap_or(f64::MAX) as i64
                });
                if let Some((id, _)) = closest_waller {
                    return *id;
                }
            } else {
                // Try attackers
                let closest_attacker = self.attack.attackers.iter().min_by_key(|(id, _)| {
                    ctx.world
                        .get_player(*id)
                        .map(|p| (p.position - ball.position.xy()).norm())
                        .unwrap_or(f64::MAX) as i64
                });
                if let Some((id, _)) = closest_attacker {
                    return *id;
                }

                // Try harrassers
                if let Some(harasser) = harasser {
                    return harasser;
                }

                // Try wallers
                let closest_waller = self.defense.wallers.iter().min_by_key(|(id, _)| {
                    ctx.world
                        .get_player(*id)
                        .map(|p| (p.position - ball.position.xy()).norm())
                        .unwrap_or(f64::MAX) as i64
                });
                if let Some((id, _)) = closest_waller {
                    return *id;
                }
            }
        }

        keeper
    }
}

impl Strategy for PlayStrategy {
    fn name(&self) -> &'static str {
        "Play"
    }

    fn update(&mut self, ctx: StrategyCtx) {
        let game_state = ctx.world.current_game_state.game_state;

        // Free kick
        if matches!(game_state, GameState::FreeKick)
            && ctx.world.current_game_state.us_operating
            && (self.last_game_state != GameState::FreeKick || self.free_kick_attacker.is_none())
        {
            // Switch to free kick strategy
            let kicker = self.get_player_for_kick(ctx.clone());
            log::info!("Free kick attacker: {:?}", kicker);
            self.free_kick_attacker = Some((kicker, FreeAttacker::new()));
        }

        // Play
        if matches!(
            game_state,
            GameState::Run | GameState::Stop | GameState::BallReplacement(_)
        ) {
            self.free_kick_attacker = None;

            self.defense.wallers.iter_mut().for_each(|w| {
                w.1.goalie_shooting(self.defense.keeper.kicking_to.is_some())
            });

            if let Some((_, harasser)) = self.defense.harasser.as_mut() {
                let attacker_positions = self
                    .attack
                    .attackers
                    .iter()
                    .map(|(id, _)| {
                        ctx.world
                            .get_player(*id)
                            .map(|p| p.position.xy())
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>();

                if let Some(pos) = attacker_positions.first() {
                    harasser.set_shooting_target(*pos);
                }
            }

            if let Some(ball) = ctx.world.ball.as_ref() {
                if let Some(id) = self.defense.keeper.kicking_to {
                    if let Some((_, attacker)) =
                        self.attack.attackers.iter_mut().find(|(i, _)| *i == id)
                    {
                        attacker.receive();
                    }
                }

                let ball_speed = ball.velocity.xy().norm();
                if ball.position.x > 0.0 && ball_speed < 500.0 && !self.attack.we_have_ball() {
                    self.attack.fetch_ball(&ctx);
                }
            }

            self.attack.update(ctx);
        }

        self.last_game_state = game_state;
    }

    fn get_role(&mut self, player_id: PlayerId, ctx: StrategyCtx) -> Option<&mut dyn Role> {
        let game_state = ctx.world.current_game_state.game_state;

        match game_state {
            dies_core::GameState::FreeKick
                if self
                    .free_kick_attacker
                    .as_ref()
                    .map(|(id, _)| *id == player_id)
                    .unwrap_or(false)
                    && ctx.world.current_game_state.us_operating =>
            {
                if let Some((_, role)) = self.free_kick_attacker.as_mut() {
                    Some(role as &mut dyn Role)
                } else {
                    None
                }
            }
            _ => self.attack.get_role(player_id, ctx).or_else(|| {
                if player_id == self.defense.keeper_id {
                    Some(&mut self.defense.keeper)
                } else if self.defense.harasser.as_ref().map(|(id, _)| *id) == Some(player_id) {
                    self.defense
                        .harasser
                        .as_mut()
                        .map(|(_, role)| role as &mut dyn Role)
                } else {
                    self.defense
                        .wallers
                        .iter_mut()
                        .find_map(|(id, role)| if *id == player_id { Some(role) } else { None })
                        .map(|role| role as &mut dyn Role)
                }
            }),
        }
    }
}
