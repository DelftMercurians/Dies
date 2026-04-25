// Exercise the SkillExecutor.PickupBall path without a strategy binary.

globalThis.scenario = {
  name: "skill_pickup_ball",
  team: "blue",
  robots: [1],
  env: "sim",
};

globalThis.run = async function run({ team, world, log }) {
  const r = team.robot(1);
  await world.addRobot({ team: "blue", id: 1, x: -500, y: 0, yaw: 0 });
  // Nudge the ball toward origin (sim-only — setBallForce applies an impulse).
  world.setBallForce({ x: 0, y: 0 });
  await sleep(200);

  try {
    await r.pickupBall({ heading: 0 });
    log.info("pickupBall succeeded");
    log.record("skill_pickup_ball", { result: "succeeded" });
  } catch (e) {
    log.warn(`pickupBall failed: ${e}`);
    log.record("skill_pickup_ball", { result: "failed", error: String(e) });
  }
};
