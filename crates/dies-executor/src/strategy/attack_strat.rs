use dies_core::{PlayerId, WorldData};

use crate::roles::{
    attacker::{Attacker, AttackerSection, AttackerState},
    waller::Waller,
    Goalkeeper, Role,
};

use super::{Strategy, StrategyCtx};

pub struct Attack {
    attackers: Vec<(PlayerId, Attacker)>,
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
                receiver.receive();
            } else {
                log::error!("Receiver not found: {:?}", receiver);
            }
        }
    }

    fn fetch_ball(&mut self, ctx: &StrategyCtx) {
        if self.we_fetching_ball() {
            return;
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
                if let Some((_, attacker)) =
                    self.attackers.iter_mut().find(|(id, _)| *id == active_attacker.id)
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

    fn get_role(&mut self, player_id: PlayerId) -> Option<&mut dyn Role> {
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
}

impl Defense {
    pub fn new(keeper_id: PlayerId) -> Self {
        Self {
            keeper_id,
            keeper: Goalkeeper::new(),
            wallers: Vec::new(),
        }
    }

    pub fn add_wallers(&mut self, player_ids: Vec<PlayerId>) {
        // -50, 50, -100, 100, etc
        let offsets = (0..player_ids.len())
            .map(|i| if i % 2 == 0 { -50.0 } else { 50.0 })
            .collect::<Vec<_>>();
        for (id, offset) in player_ids.iter().zip(offsets) {
            self.wallers.push((*id, Waller::new(offset)));
        }
    }

    pub fn update(&mut self, ctx: StrategyCtx) {}
}

pub struct PlayStrategy {
    pub attack: Attack,
    pub defense: Defense,
}

impl PlayStrategy {
    pub fn new(keeper_id: PlayerId) -> Self {
        Self {
            attack: Attack::new(),
            defense: Defense::new(keeper_id),
        }
    }
}

impl Strategy for PlayStrategy {
    fn name(&self) -> &'static str {
        "Play"
    }

    fn update(&mut self, ctx: StrategyCtx) {
        // let waller_ids = self
        //     .defense
        //     .wallers
        //     .iter()
        //     .map(|(id, _)| *id)
        //     .collect::<Vec<_>>();
        // let attacker_ids = self
        //     .attack
        //     .attackers
        //     .iter()
        //     .map(|(id, _)| *id)
        //     .collect::<Vec<_>>();
        // for player in ctx.world.own_players.iter() {
        //     if player.id == self.defense.keeper_id
        //         || waller_ids.contains(&player.id)
        //         || attacker_ids.contains(&player.id)
        //     {
        //         continue;
        //     }

        //     if waller_ids.len() < 2 {
        //         self.defense.add_wallers(player.id)
        //     }player_ids
        // }

        if let Some(ball) = ctx.world.ball.as_ref() {
            let ball_speed = ball.velocity.xy().norm();
            if ball_dist_to_closest_enemy(ctx.world) > 300.0
                // && ball.position.x < 0.0
                && ball_speed < 300.0
                && !self.attack.we_have_ball()
            {
                self.attack.fetch_ball(&ctx);
            }

            if ball.position.x < -5000.0
                // && ball.position.x < 0.0
                && ball_speed < 300.0
                && !self.attack.we_have_ball()
            {
                self.attack.fetch_ball(&ctx);
            }

            // if ball.position.x > 0.0
            //     && ball_dist_to_closest_enemy(world) > 150.0
            //     && ball_speed < 300.0
            // {}
        }

        self.attack.update(ctx);
    }

    fn get_role(&mut self, player_id: PlayerId) -> Option<&mut dyn Role> {
        self.attack.get_role(player_id).or_else(|| {
            if player_id == self.defense.keeper_id {
                Some(&mut self.defense.keeper)
            } else {
                self.defense
                    .wallers
                    .iter_mut()
                    .find_map(|(id, role)| if *id == player_id { Some(role) } else { None })
                    .map(|role| role as &mut dyn Role)
            }
        })
    }
}

fn ball_dist_to_closest_enemy(world: &WorldData) -> f64 {
    let ball_pos = world.ball.as_ref().unwrap().position.xy();
    world
        .opp_players
        .iter()
        .map(|p| (p.position - ball_pos).norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f64::MAX)
}

fn ball_dist_to_closest_waller(world: &WorldData, wallers: Vec<PlayerId>) -> f64 {
    let ball_pos = world.ball.as_ref().unwrap().position.xy();
    world
        .own_players
        .iter()
        .filter(|p| wallers.contains(&p.id))
        .map(|p| (p.position - ball_pos).norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f64::MAX)
}
