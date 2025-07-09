import "game_modes/play" as play;
import "game_modes/kickoff" as kickoff;
import "game_modes/penalty" as penalty;


fn build_play_bt(player_id) {
    return play::build_play_bt(player_id);
}

fn build_kickoff_bt(player_id) {
    return kickoff::build_kickoff_bt(player_id);
}

fn build_penalty_bt(player_id) {
    return penalty::build_penalty_bt(player_id);
}