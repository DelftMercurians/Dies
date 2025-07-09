import "shared/utilities" as util;
import "shared/situations" as sit;
import "shared/subtrees" as trees;

fn build_play_bt(player_id) {
    // // Goalkeeper behavior (player 0)
    if player_id.to_string() == "0" {
        return trees::build_goalkeeper_behavior();
    }
    
    // Field players use dynamic role assignment
    return ScoringSelect([
        #{ 
            node: trees::build_attacker_behavior(), 
            scorer: |s| util::score_as_attacker(s) 
        },
        #{ 
            node: trees::build_supporter_behavior(), 
            scorer: |s| util::score_as_supporter(s) 
        },
        #{ 
            node: trees::build_defender_behavior(), 
            scorer: |s| util::score_as_defender(s) 
        }
    ], 0.15, "Dynamic Role Assignment");
} 