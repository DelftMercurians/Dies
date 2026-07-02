TODO

- yello card removal, robot not removed from start control, oscillating
- strategy: dumb down, shots on goal, no passing, strong defense
- pickup
  - scenario: one for each robot
  - once we have data, tune min speed and gain
- set kick speeds per robot id

- check logs pickup for p3 getting stuck

TESTING

- dumbed strategy
- handicaps
- pikcup
- defense
- yellow card pull

TODO

```
Step::skill("shoot at goal", r, move |h| {
                    h.handle_ball(
                        BallAction::Shoot {
                            target: Vector2::new(-4500.0, 0.0),
                        },
                        AcquirePosition::Fastest,
                    );
                })
```

dribbler turns off mid skill?

keeper
HandleBallFailed
hold: acquiring · bail: defense area
