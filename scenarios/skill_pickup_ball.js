globalThis.scenario = {
  name: "skill_pickup_ball",
  team: "blue",
  robots: [1],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const R1_ID = 5;
  const R2_ID = 1;
  let r1 = team.robot(R1_ID);
  let r2 = team.robot(R2_ID);

  try {
    while (true) {
      await Promise.all([
        r1.moveTo({ x: -1500, y: 0 }),
        r2.moveTo({ x: 1500, y: 0 })
      ]);
      await sleep(1000);
      log.info("attempting pickupBall...");


      log.info(`ball position: ${JSON.stringify(world.ball().position)}`);
      log.info(`robot 2 position: ${JSON.stringify(world.robot(R2_ID)?.position)}`);
      log.info(`robot 1 position: ${JSON.stringify(world.robot(R1_ID)?.position)}`);
      let { x: ballX, y: ballY } = world.ball().position;
      let r2Pos = world.robot(R2_ID)?.position;

      // angle from ball to r2
      let shotHeading = Math.atan2(r2Pos.y - ballY, r2Pos.x - ballX);
      const receivePromise = r2.receive({
        from: { x: ballX, y: ballY },
        target: { x: r2Pos.x, y: r2Pos.y },
        captureLimit: 1000,
      })
      await r1.pickupBall({ heading: shotHeading });
      r2Pos = world.robot(R2_ID)?.position;
      ballX = world.ball().position.x;
      ballY = world.ball().position.y;

      log.info("pickupBall succeeded");

      const r1Pos = world.robot(R1_ID)?.position;

      await r1.shoot({ target: { tx: r2Pos.x, ty: r2Pos.y } });
      await receivePromise;

      await sleep(2000);
      [r1, r2] = [r2, r1];
    }
  } catch (e) {
    log.warn(`pickupBall failed: ${e}`);
    log.record("skill_pickup_ball", { result: "failed", error: String(e) });
  }
};
