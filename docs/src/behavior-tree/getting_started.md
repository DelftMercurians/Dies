# Getting Started: Your First Behavior Tree

This guide will walk you through creating a simple behavior tree script using the modern **role assignment system**.

## Entry Point

There is a single `main(game)` function per strategy that adds roles dynamically using the game context:

```rhai
// strategies/main.rhai

fn main(game) {
    // The game parameter provides context about the current situation
    print(`Game state: ${game.game_state}, Players: ${game.num_own_players}`);

    // Add roles using the fluent API - no return value needed
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)
        .require(|s| s.player_id == 0)
        .behavior(|| build_goalkeeper_tree())
        .build();

    game.add_role("attacker")
        .max(1)
        .score(|s| score_attacker(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| build_attacker_tree())
        .build();
}
```

## Example 1: Basic Role Assignment

Let's start with a simple role assignment that creates basic team roles:

```rhai
// strategies/main.rhai
import "shared/utilities" as util;
import "shared/subtrees" as trees;

fn main(game) {
    // Always need exactly one goalkeeper
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)
        .require(|s| s.player_id == 0)  // Only robot 0 can be goalkeeper
        .behavior(|| trees::build_goalkeeper_tree())
        .build();

    // One main attacker
    game.add_role("attacker")
        .max(1)
        .score(|s| util::score_attacker(s))  // Closest to ball gets highest score
        .exclude(|s| s.player_id == 0)      // Exclude goalkeeper
        .behavior(|| trees::build_attacker_tree())
        .build();

    // Remaining robots become defenders
    game.add_role("defender")
        .min(1)  // At least one defender
        .score(|s| util::score_defender(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| trees::build_defender_tree())
        .build();
}
```

## Example 2: Dynamic Roles Based on Game State

Here's how to adapt roles based on the game situation:

```rhai
// strategies/main.rhai
import "shared/utilities" as util;
import "shared/subtrees" as trees;

fn main(game) {
    // Add base roles
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)
        .require(|s| s.player_id == 0)
        .behavior(|| trees::build_goalkeeper_tree())
        .build();

    game.add_role("defender")
        .min(1)
        .max(2)
        .score(|s| util::score_defender(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| trees::build_defender_tree())
        .build();

    // Add special roles based on game state
    switch game.game_state {
        "FreeKick" => {
            print("Adding free kicker for free kick situation");
            game.add_role("free_kicker")
                .count(1)
                .score(|s| util::score_free_kicker(s))
                .exclude(|s| s.player_id == 0)
                .behavior(|| trees::build_free_kicker_tree())
                .build();
        },
        "PenaltyKick" => {
            print("Adding penalty taker for penalty situation");
            game.add_role("penalty_taker")
                .count(1)
                .score(|s| util::score_penalty_taker(s))
                .exclude(|s| s.player_id == 0)
                .behavior(|| trees::build_penalty_taker_tree())
                .build();
        },
        _ => {
            // Normal play - add regular attacker
            game.add_role("attacker")
                .max(1)
                .score(|s| util::score_attacker(s))
                .exclude(|s| s.player_id == 0)
                .behavior(|| trees::build_attacker_tree())
                .build();
        }
    }
}
```

## Example 3: Adapting to Player Count

You can also adapt your strategy based on how many players are available:

```rhai
fn main(game) {
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)
        .require(|s| s.player_id == 0)
        .behavior(|| trees::build_goalkeeper_tree())
        .build();

    // Adapt strategy based on available players
    if game.num_own_players >= 3 {
        // Full team - can have specialized roles
        game.add_role("attacker")
            .max(1)
            .score(|s| util::score_attacker(s))
            .exclude(|s| s.player_id == 0)
            .behavior(|| trees::build_attacker_tree())
            .build();

        game.add_role("defender")
            .min(1)
            .score(|s| util::score_defender(s))
            .exclude(|s| s.player_id == 0)
            .behavior(|| trees::build_defender_tree())
            .build();
    } else {
        // Limited players - use flexible support role
        game.add_role("support")
            .min(1)
            .score(|s| util::score_support(s))
            .exclude(|s| s.player_id == 0)
            .behavior(|| trees::build_support_tree())
            .build();
    }
}
```
