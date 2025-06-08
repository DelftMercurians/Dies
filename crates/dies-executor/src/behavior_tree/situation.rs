use super::bt_callback::BtCallback;
use super::bt_core::RobotSituation;

#[derive(Clone)]
enum SituationCondition {
    Fn(BtCallback<bool>),
    And(Box<SituationCondition>, Box<SituationCondition>),
    Or(Box<SituationCondition>, Box<SituationCondition>),
    Not(Box<SituationCondition>),
}

impl SituationCondition {
    pub fn check(&self, situation: &RobotSituation) -> bool {
        match self {
            SituationCondition::Fn(f) => f.call(situation).unwrap_or(false),
            SituationCondition::And(a, b) => a.check(situation) && b.check(situation),
            SituationCondition::Or(a, b) => a.check(situation) || b.check(situation),
            SituationCondition::Not(a) => !a.check(situation),
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

    pub fn check(&self, situation: &RobotSituation) -> bool {
        self.condition.check(situation)
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
