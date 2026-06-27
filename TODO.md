Skills

- ball handling
  - pickup ball
    - with heading, or quickest appraoch (aiming handed over to shoot skill)
    - todo: remove intercept arm; approach only when ball is in the dribbler cone; tune ball avoidance care
    - maybe: allow kicking right after approach?
  - dribblet shoot
    - aim first (either turn with ball or orbit) then kick either on correct heading or fallback trigger
    - optionally allow moving to target with dribbling (extra)
    - todo: implement and test orbit; switch between turn and orbit based on available space;
  - reflex shoot
    - maintain heading, kick on breakbream trigger
    - tof repositioning (extra)
    - todo: test, map out angle response
  - receive pass
    - move to intercept, track ball, cushion receive
- movement
-

Strategy

- amping up aggressions
  - Grab-first pikcup - no heading angle just get ball at the nearest point
  - dedicated snatch skill - use the rotating trick from the bots
- more robust ball progression
  - ~~explicit prepare passing receiver under planner control~~
- more structured defense
  - smarter selection of which defenders can be commited to the plan - use centralized metric computed by the formation
  - figure out why clustering happens and solve
    - too many shadow roles / not good spacing
    - ~~marking players need to be between the ball and the attacker - not between the goal and the attacker~~
    - shadow/mark too close to bal pickup
    - limit shadows to 2 - push others to mark
      - one high prio shhadow that covers a tight arc around the defense area
      - on auxiliary shadow that steps in when threat increases to mazimise coverage
    - ~~need to coordinate with planner allocated robot during snat/pickup~~
- ball pickup and snatch
  - when opponent is also attempting to pickup the ball we can edn up in a deadlock - need to think about the right approah
- keeper
  - postiion control flags: tune up aggresivness (higher break gain, higher kp), disable orca and planner
  - move on small arc around the goal mouth
  - kick ball out of the goal area when possible

UI

- team inspector:
  - show critial hardware statuses
  - add audible alarms
  - add active skill + skill state desc
- add player ids on field renderer
- render current plan in sidebar
