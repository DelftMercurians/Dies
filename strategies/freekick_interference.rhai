fn build_free_kick_interference_tree() {
    return GoToPosition(
        |s| calculate_free_kick_defense_position(s),
        #{ heading: |s| get_defender_heading(s) },
        "Free kick defense"
    );
}

fn score_free_kick_interference(s) {
    let score = 70.0;
    
    // Prefer robots that can position between ball and our goal
    if s.world.ball != () {
        let ball_pos = s.world.ball.position;
        let goal_pos = s.get_own_goal_position();
        let my_pos = s.player.position;
        
        // Calculate positioning score
        let ball_to_goal = goal_pos - ball_pos;
        let ball_to_me = my_pos - ball_pos;
        let projection = (ball_to_me.x * ball_to_goal.x + ball_to_me.y * ball_to_goal.y) / ball_to_goal.norm();
        let projection_ratio = projection / ball_to_goal.norm();
        
        if projection_ratio > 0.2 && projection_ratio < 0.8 {
            score += 20.0;
        }
    }
    
    return score;
}

fn calculate_free_kick_defense_position(s) {
    if s.world.ball == () {
        return s.get_own_goal_position() + vec2(1000.0, 0.0);
    }
    
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    let ball_to_goal = (goal_pos - ball_pos).unit();
    
    // Position 700mm from ball toward goal (SSL rule: 500mm minimum)
    return ball_pos + ball_to_goal * 700.0;
}

fn get_defender_heading(s) {
    if s.world.ball == () { return 0.0; }
    let ball_pos = s.world.ball.position;
    let my_pos = s.player.position;
    return (ball_pos - my_pos).angle_to(vec2(1.0, 0.0));
}