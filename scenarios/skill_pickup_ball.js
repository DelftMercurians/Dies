// Exercise the SkillExecutor.PickupBall path without a strategy binary.

globalThis.scenario = {
  name: "skill_pickup_ball",
  team: "blue",
  robots: [1],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const r = team.robot(5);
  // await r.kick({ force: 6000 });
  // await sleep(200);
  await r.moveTo({ x: -1000, y: 0, yaw: 0 }, { tolMm: 50 });

  try {
    await r.pickupBall({ heading: 0 });
    await r.kick({ force: 6000 });
    await sleep(200);
    log.info("pickupBall succeeded");
    log.record("skill_pickup_ball", { result: "succeeded" });
  } catch (e) {
    log.warn(`pickupBall failed: ${e}`);
    log.record("skill_pickup_ball", { result: "failed", error: String(e) });
  }
};
