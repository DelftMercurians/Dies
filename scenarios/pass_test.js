// Two-robot pass test scenario
// Passer: picks up ball at origin, dribbles, and shoots to receiver
// Receiver: moves to (3000, 0), receives the pass with cushioning

globalThis.scenario = {
  name: "pass_test",
  team: "blue",
  robots: [0, 5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const PASSER_ID = 0;
  const RECEIVER_ID = 5;
  
  const passer = team.robot(PASSER_ID);
  const receiver = team.robot(RECEIVER_ID);

  try {
    log.info("=== Pass Test Scenario ===");
    log.info("Passer: picks up ball and dribbles to origin");
    log.info("Receiver: moves to (3000, 0) and receives pass");

    // Initial setup: position robots
    log.info("Positioning robots...");
    await Promise.all([
      passer.moveTo({ x: -1000, y: 0, yaw: 0 }),
      receiver.moveTo({ x: 3000, y: 0, yaw: Math.PI })
    ]);

    log.info("Starting pass sequence...");

    // Phase 1: Passer picks up ball and dribbles
    const passerTask = (async () => {
      try {
        log.info("Passer: picking up ball at heading 0°");
        await passer.pickupBall({ heading: 0 });
        log.info("Passer: ball picked up, starting dribble");
        
        // Dribble to origin with rotate-around-ball mode (with_ball: false)
        await passer.dribble({
          target: { x: 0, y: 0 },
          heading: 0,
          withBall: false
        });
        log.info("Passer: dribble complete, shooting");
        
        await passer.shoot({ target: { x: 3000, y: 0 } });
        log.info("Passer: shot complete");
      } catch (e) {
        log.error(`Passer error: ${e}`);
      }
    })();

    // Phase 2: Receiver waits and receives pass
    const receiverTask = (async () => {
      try {
        log.info("Receiver: moving to final position (3000, 0)");
        await receiver.moveTo({ x: 3000, y: 0, yaw: Math.PI });
        
        log.info("Receiver: waiting for pass from (0, 0)");
        await receiver.receive({
          from: { x: 0, y: 0 },
          target: { x: 3000, y: 0 },
          captureLimit: 2000,
          cushion: true
        });
        log.info("Receiver: pass received!");
      } catch (e) {
        log.error(`Receiver error: ${e}`);
      }
    })();

    // Run both sequences in parallel
    await Promise.all([passerTask, receiverTask]);

    log.info("=== Pass Test Complete ===");

  } catch (e) {
    log.error(`Scenario error: ${e}`);
  }
}
