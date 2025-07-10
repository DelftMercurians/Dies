import "shared/utilities" as util;
import "shared/situations" as sit;
import "shared/subtrees" as trees;

fn main() {
    return AssignRoles([
        Role("goalkeeper")
            .count(1)
            .score(|s| 100.0)
            .require(|s| s.player_id == 0)
            .behavior(|| trees::build_goalkeeper_tree())
            .build(),

        Role("striker")
            .max(1)
            .score(|s| util::score_striker(s))
            .exclude(|s| s.player_id == 0)
            .behavior(|| trees::build_striker_tree())
            .build(),

        Role("defender")
            .min(1)
            .max(2)
            .score(|s| util::score_defender(s))
            .exclude(|s| s.player_id == 0)
            .behavior(|| trees::build_defender_tree())
            .build(),

        Role("support")
            .min(1)
            .score(|s| util::score_support(s))
            .exclude(|s| s.player_id == 0)
            .behavior(|| trees::build_support_tree())
            .build()
    ]);
}