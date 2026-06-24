globalThis.scenario = {
  name: "skill_dribble_shoot",
  team: "blue",
  robots: [1],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const r1 = team.robot(0);

  try {
    // Deterministic setup: robot behind the ball, both on the +x axis.
    // r1.teleport({ x: -300, y: 0, yaw: 0 });
    // world.setBall({ x: 0, y: 0 });
    // await sleep(500);

    while (true) {
      // Capture facing +x, then aim 90° away by orbiting the ball.
      await r1.pickupBall({ heading: Math.PI });
      // log.info("ball captured, starting dribbleShoot");
      await r1.dribbleShoot({ heading: 0 });
      log.info("dribbleShoot succeeded");
    }
  } catch (e) {
    log.warn(`${e}`);
  }
};
