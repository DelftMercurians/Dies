# Dies Strategy Scripts

This directory contains the modular behavior tree strategy scripts for the Dies robot soccer framework. The scripts are organized using Rhai's module system for better maintainability and organization.

## Directory Structure

```
strategies/
├── main.rhai              # Main entry point - imports and combines all modules
├── shared/                # Shared components used across game modes
│   ├── utilities.rhai     # Helper functions, constants, and positioning logic
│   ├── situations.rhai    # Condition functions for Guards
│   └── subtrees.rhai      # Reusable behavior tree components
└── game_modes/            # Game mode specific behaviors
    ├── play.rhai          # Normal gameplay including free kicks
    ├── kickoff.rhai       # Kickoff situations (both offensive and defensive)
    └── penalty.rhai       # Penalty situations (both attacking and defending)
```

## Module Organization

### Main Entry Point (`main.rhai`)

- Imports all game mode modules
- Provides the main entry points called by the Dies executor:
  - `build_play_bt(player_id)` - Normal gameplay
  - `build_kickoff_bt(player_id)` - Kickoff situations
  - `build_penalty_bt(player_id)` - Penalty situations
  - `build_player_bt(player_id)` - Legacy entry point (defaults to play mode)

### Shared Modules (`shared/`)

#### `utilities.rhai`

Contains helper functions and constants:

- Goal positions (`get_opponent_goal()`, `get_own_goal()`)
- Ball position helpers (`get_ball_pos()`, `get_ball_velocity()`)
- Distance calculations (`distance_to_ball()`, `distance_to_own_goal()`)
- Role scoring functions (`score_as_attacker()`, `score_as_supporter()`, `score_as_defender()`)
- Dynamic positioning functions (`get_supporter_pos()`, `get_defender_pos()`, etc.)

#### `situations.rhai`

Contains condition functions for Guard nodes:

- Ball possession checks (`i_have_ball()`)
- Ball position checks (`ball_in_our_half()`, `ball_in_opponent_half()`)
- Player role checks (`is_goalkeeper()`)
- Distance-based conditions (`close_to_ball()`, `far_from_ball()`)

#### `subtrees.rhai`

Contains reusable behavior tree components:

- Player role behaviors (`build_attacker_behavior()`, `build_supporter_behavior()`, etc.)
- Game mode specific behaviors (`build_kickoff_kicker_behavior()`, `build_penalty_taker_behavior()`, etc.)

### Game Mode Modules (`game_modes/`)

#### `play.rhai`

- Normal gameplay behavior including free kicks
- Dynamic role assignment using ScoringSelect
- Goalkeeper and field player behaviors

#### `kickoff.rhai`

- Rule-compliant kickoff behavior
- Kicker selection using semaphores
- Supporter positioning in own half

#### `penalty.rhai`

- Rule-compliant penalty behavior
- Penalty taker selection using semaphores
- Goalkeeper positioning on goal line
- Support player positioning behind ball

## Usage

The modular system allows for:

1. **Better Organization**: Related functions are grouped together
2. **Code Reuse**: Shared behaviors and utilities can be used across game modes
3. **Easier Maintenance**: Changes to specific game modes don't affect others
4. **Clearer Dependencies**: Import statements make dependencies explicit

### Example Usage in Scripts

```rust
// Import modules at the top of your script
import "shared/utilities" as util;
import "shared/situations" as sit;
import "shared/subtrees" as trees;

// Use imported functions with module prefixes
fn my_behavior(player_id) {
    return Guard(sit::i_have_ball,
        GoToPosition(util::get_opponent_goal(), #{}, "Go to goal"),
        "Have ball?"
    );
}
```

## Extending the System

To add new behaviors:

1. **For shared functionality**: Add to appropriate `shared/` module
2. **For game mode specific behavior**: Add to or create new game mode module
3. **For new game modes**: Create new file in `game_modes/` and import in `main.rhai`

All strategies are designed to be rule-compliant with RoboCup SSL 2025 rules.
