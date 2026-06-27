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
  - **explicit prepare passing receiver under planner control**
- more structured defense
  - smarter selection of which defenders can be commited to the plan - use centralized metric computed by the formation
  - figure out why clustering happens and solve
    - too many shadow roles / not good spacing
    - ~~marking players need to be between the ball and the attacker - not between the goal and the attacker~~
    - shadow/mark too close to bal pickup
    - limit shadows to 2 - push others to mark
      - one high prio shhadow that covers a tight arc around the defense area
      - on auxiliary shadow that steps in when threat increases to mazimise coverage
    - **need to coordinate with planner allocated robot during snat/pickup** - should be counted as a mark/shadow
- keeper
  - postiion control flags: tune up aggresivness (higher break gain, higher kp), disable orca and planner
  - move on small arc around the goal mouth

UI

- change layout to this:

```json
{
  "grid": {
    "root": {
      "type": "branch",
      "data": [
        {
          "type": "leaf",
          "data": {
            "views": ["field"],
            "activeView": "field",
            "id": "1",
            "hideHeader": true
          },
          "size": 1243
        },
        {
          "type": "branch",
          "data": [
            {
              "type": "leaf",
              "data": {
                "views": [
                  "player-inspector",
                  "game-controller",
                  "strategy",
                  "debug-layers"
                ],
                "activeView": "debug-layers",
                "id": "4"
              },
              "size": 0
            },
            {
              "type": "leaf",
              "data": {
                "views": ["console", "settings", "scenario"],
                "activeView": "settings",
                "id": "3"
              },
              "size": 900
            }
          ],
          "size": 413
        }
      ],
      "size": 900
    },
    "width": 1656,
    "height": 900,
    "orientation": "HORIZONTAL"
  },
  "panels": {
    "field": {
      "id": "field",
      "contentComponent": "field",
      "title": "FIELD"
    },
    "console": {
      "id": "console",
      "contentComponent": "console",
      "title": "CONSOLE"
    },
    "settings": {
      "id": "settings",
      "contentComponent": "settings",
      "title": "SETTINGS"
    },
    "scenario": {
      "id": "scenario",
      "contentComponent": "scenario",
      "title": "SCENARIO"
    },
    "player-inspector": {
      "id": "player-inspector",
      "contentComponent": "player-inspector",
      "title": "INSPECTOR"
    },
    "game-controller": {
      "id": "game-controller",
      "contentComponent": "game-controller",
      "title": "GAME CTRL"
    },
    "strategy": {
      "id": "strategy",
      "contentComponent": "strategy",
      "title": "STRATEGY"
    },
    "debug-layers": {
      "id": "debug-layers",
      "contentComponent": "debug-layers",
      "title": "DEBUG"
    }
  },
  "activeGroup": "3"
}
```

- team inspector:
  - show critial hardware statuses
  - add audible alarms
  - add active skill + skill state desc
- add player ids on field renderer
- render current plan in sidebar
