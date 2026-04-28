UI

- [ ] Add keyboard shorcuts to UI
- [ ] Make it easier to access debug values, remove mini scrollbars
- [ ] Clean up debug tags, make sure everything is properly grouped
- [ ] Sim: make it possible to directly move the ball or robots by dragging them
- [ ] More knobs for togglging GC, obstacle avoidance, white/blacklisting robots, etc

iLQR

- [ ] Obstacle avoidance
- [ ] Visualize cost function and dynamics, look at parameters
- [ ] Come up with tuning procedure

Refactoring

- [ ] Clean up executor/strategy/test-driver/controller path. minimize in between layers
- [ ] Clean up test driver JS bridge, add macros for defining exposed functions, generate JSDoc or TS types
- [ ] Clean up iLQR integration. it shouldnt be an override

Fixes

- [ ] Test and fix ghost rejection, make sure robots not in vision are properly handled
- [ ] Look into fixed control loop rate - should we enforce it?

Skills

- [ ] Come up with a way to coordinate passes on the skill level
- [ ] Pickup ball interception
