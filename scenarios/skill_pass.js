// Smoke test for the joint pass coordinator (ctx.pass / team.pass).
//
// Each drill resets the ball + robots to a known pose, then exercises one
// aspect of the atomic pass: the happy path, a clean mid-pass cancel, and the
// ball-not-secured failure. Assertions check the typed PassFailure reason
// surfaced by the rejected promise.
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

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // Wait until vision has both robots and a ball.
  await waitUntil(
    () => world.ball() && world.robot(PASSER) && world.robot(RECEIVER),
    { pollMs: 50, timeoutMs: 5000 }
  );

  // Reset to a clean pose: ball at center, passer just behind it (aimed at the
  // receiver downfield), receiver waiting downfield. The ball is teleported back
  // so each drill starts from the same geometry instead of wherever the previous
  // drill left it.
  async function reset({ passerX = -350, passerY = 0 } = {}) {
    world.setBall({ x: 0, y: 0 });
    await passer.teleport({ x: passerX, y: passerY, yaw: 0 });
    await receiver.teleport({ x: 1500, y: 0, yaw: Math.PI });
    await sleep(400);
  }

  function reasonOf(e) {
    return String(e && e.message ? e.message : e);
  }

  // --- Drill 1: happy path -------------------------------------------------
  log.info("=== Drill 1: happy path ===");
  await reset();
  try {
    await team.pass({ passer: PASSER, receiver: RECEIVER });
    log.info("drill1 PASS: pass succeeded");
  } catch (e) {
    log.error(`drill1 FAIL: expected success, got ${reasonOf(e)}`);
  }
  await sleep(500);

  // --- Drill 2: cancel mid-pass (clean release / PartnerLeft) ---------------
  log.info("=== Drill 2: cancel mid-pass ===");
  await reset();
  {
    // Aim far downfield on the approach line so the receiver has to travel to
    // get ready — this keeps the pass in its cancellable Setup phase long
    // enough for the passer to actually secure the ball before we cancel.
    const p = team.pass({
      passer: PASSER,
      receiver: RECEIVER,
      targetHint: { x: 3000, y: 0 },
    });
    // Let the passer actually engage the ball first, so the cancel is a real
    // mid-pass abort rather than a no-op on an idle robot.
    await waitUntil(
      () => {
        const pp = world.robot(PASSER);
        const b = world.ball();
        return pp && b && dist(pp.position, b.position) < 160;
      },
      { pollMs: 50, timeoutMs: 3000 }
    );
    // Reassign the receiver to something else — this replaces its pass slot,
    // orphaning the passer's half.
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
    // Passer starts beyond the secure range: the weak Secure phase never chases
    // a loose ball, so it must decline and fail cleanly with BallLost.
    world.setBall({ x: 0, y: 0 });
    await passer.teleport({ x: 1000, y: 0, yaw: 0 });
    await receiver.teleport({ x: 1500, y: 0, yaw: Math.PI });
    await sleep(400);
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

  // Stop the skill on both robots so they don't keep running their last skill
  // (drill 2's goToPos) after the scenario ends.
  passer.skillStop();
  receiver.skillStop();
  await sleep(200);
  log.info("skill_pass scenario complete");
};
