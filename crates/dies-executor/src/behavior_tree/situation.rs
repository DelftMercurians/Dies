use super::bt_callback::BtCallback;
use super::bt_core::RobotSituation;
use rhai::Engine;

#[derive(Clone)]
enum SituationCondition {
    Fn(BtCallback<bool>),
    And(Box<SituationCondition>, Box<SituationCondition>),
    Or(Box<SituationCondition>, Box<SituationCondition>),
    Not(Box<SituationCondition>),
}

impl SituationCondition {
    pub fn check(&self, situation: &RobotSituation, engine: &Engine) -> bool {
        match self {
            SituationCondition::Fn(f) => f.call(situation, engine).unwrap_or(false),
            SituationCondition::And(a, b) => {
                a.check(situation, engine) && b.check(situation, engine)
            }
            SituationCondition::Or(a, b) => {
                a.check(situation, engine) || b.check(situation, engine)
            }
            SituationCondition::Not(a) => !a.check(situation, engine),
        }
    }
}

#[derive(Clone)]
pub struct Situation {
    condition: SituationCondition,
    description: String,
}

impl Situation {
    pub fn new_fn(condition: BtCallback<bool>, description: &str) -> Self {
        Self {
            condition: SituationCondition::Fn(condition),
            description: description.to_string(),
        }
    }

    pub fn check(&self, situation: &RobotSituation, engine: &Engine) -> bool {
        self.condition.check(situation, engine)
    }

    pub fn and(self, other: Situation) -> Self {
        let desc = format!("({}) AND ({})", self.description, other.description);
        Self {
            condition: SituationCondition::And(Box::new(self.condition), Box::new(other.condition)),
            description: desc,
        }
    }

    pub fn or(self, other: Situation) -> Self {
        let desc = format!("({}) OR ({})", self.description, other.description);
        Self {
            condition: SituationCondition::Or(Box::new(self.condition), Box::new(other.condition)),
            description: desc,
        }
    }

    pub fn not(self) -> Self {
        let desc = format!("NOT ({})", self.description);
        Self {
            condition: SituationCondition::Not(Box::new(self.condition)),
            description: desc,
        }
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}
