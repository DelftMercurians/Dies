### TODOs:
**Important:**
[ ] - final scoring rules based on exact binary heuristics (e.g. if there is a collision - the trajectory is forcefully rejected)
[ ] - continuity in control cost: prefer the control that is similar to the old one, to avoid self-reinforcing oscillatory loops
[ ] - benchmark the MPC setup based on some scoring rules and pairwise wins vs old version
[ ] - delay handling
**Mid:**
[ ] - vmap over multiple futures, take expectation when scoring trajectories - this won't affect speed, but improves robustness
[ ] - auto-calibrate all the parameters to choose the optimal MPC setup according to the scoring rules, time-limit: 1 seconds per configuration, hill-climbing with vmap
**Meh:**
[ ] - formatting, ruff, decent pyproject
[ ] - typing
[ ] - actions
[ ] - publish on pypi :)
