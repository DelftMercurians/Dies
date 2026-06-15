// Smoke test for the joint pass coordinator (ctx.pass / team.pass).
//
// Exercises the atomic pass and its clean-release failure modes. Each drill
// teleports the robots into a known pose first; assertions check the typed
// PassFailure reason surfaced by the rejected promise.
//
// Run: just test-sim skill_pass

globalThis.scenario = {
  name: "skill_pass",
  team: "blue",
  robots: [0, 1],
  env: "sim",
};

globalThis.run = async function run({ team, world, log }) {
  const PASSER = 0;
  const RECEIVER = 1;

  const passer = team.robot(PASSER);
  const receiver = team.robot(RECEIVER);

  // Wait until vision has both robots and a ball.
  await waitUntil(
    () => world.ball() && world.robot(PASSER) && world.robot(RECEIVER),
    { pollMs: 50, timeoutMs: 5000 }
  );

  // Teleport the passer just behind the ball (opposite the receiver) so the weak
  // Secure phase can pick it up, and the receiver downfield on the +x side.
  async function setupForPass() {
    const b = world.ball().position;
    await passer.teleport({ x: b.x - 350, y: b.y, yaw: 0 });
    await receiver.teleport({ x: 1500, y: 0, yaw: Math.PI });
    await sleep(300);
  }

  function reasonOf(e) {
    return String(e && e.message ? e.message : e);
  }

  // --- Drill 1: happy path -------------------------------------------------
  log.info("=== Drill 1: happy path ===");
  await setupForPass();
  try {
    await team.pass({ passer: PASSER, receiver: RECEIVER });
    log.info("drill1 PASS: pass succeeded");
  } catch (e) {
    log.error(`drill1 FAIL: expected success, got ${reasonOf(e)}`);
  }
  await sleep(500);

  // --- Drill 2: cancel mid-pass (clean release / PartnerLeft) ---------------
  log.info("=== Drill 2: cancel mid-pass ===");
  await setupForPass();
  {
    const p = team.pass({ passer: PASSER, receiver: RECEIVER });
    // Reassign the receiver to something else shortly after starting — this
    // replaces its pass slot, orphaning the passer's half.
    await sleep(300);
    receiver.goToPos({ x: -2000, y: 2000 });
    try {
      await p;
      log.error("drill2 FAIL: expected the pass to be cancelled");
    } catch (e) {
      const r = reasonOf(e);
      if (r.includes("PartnerLeft")) {
        log.info("drill2 PASS: cancelled with PartnerLeft");
      } else {
        log.warn(`drill2 UNEXPECTED reason: ${r}`);
      }
    }
    // Both robots must be immediately commandable again.
    await Promise.all([
      passer.goToPos({ x: -2000, y: -2000 }),
      receiver.goToPos({ x: -2000, y: 1500 }),
    ]);
    log.info("drill2: both robots commandable after cancel");
  }
  await sleep(300);

  // --- Drill 3: ball not secured (BallLost) ---------------------------------
  log.info("=== Drill 3: ball not secured ===");
  {
    // Put the passer far from the ball so Secure fails immediately.
    const b = world.ball().position;
    await passer.teleport({ x: b.x + 3000, y: b.y + 2000, yaw: 0 });
    await receiver.teleport({ x: 1500, y: 0, yaw: Math.PI });
    await sleep(300);
    try {
      await team.pass({ passer: PASSER, receiver: RECEIVER });
      log.error("drill3 FAIL: expected BallLost");
    } catch (e) {
      const r = reasonOf(e);
      if (r.includes("BallLost")) {
        log.info("drill3 PASS: failed with BallLost");
      } else {
        log.warn(`drill3 UNEXPECTED reason: ${r}`);
      }
    }
  }

  log.info("skill_pass scenario complete");
};
