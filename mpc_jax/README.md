### TODOs:
**Important:**
[ ] - Add the log of the MPC outputs + replay ability (so that I can have like a super long video file that would store the recording of all the candidate trajectories, optimized trajectories and the solution)
[ ] - MPC mismatch score: the difference between the "expected" position of the robot during the next step, and the actual position during the next step
[ ] - Add field constraints
[ ] - Precompile all the variants of the thing before start (maybe use full cache? if it works..)
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
