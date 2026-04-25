// Step response for motion-control tuning: robot is commanded to a point,
// position and velocity are captured for the duration, samples recorded.

globalThis.scenario = {
  name: "mpc_step_response",
  team: "blue",
  robots: [1],
  env: "sim",
};

globalThis.run = async function run({ team, world, log }) {
  const r = team.robot(1);
  await world.addRobot({ team: "blue", id: 1, x: 0, y: 0, yaw: 0 });
  await r.waitStopped({ timeoutMs: 2000 });

  // Kick off a step by asking moveTo to go to (1500, 0) — the resolution
  // condition is "within tol and stopped". Meanwhile capture for 4s.
  r.setLocalVelocity({ x: 0, y: 0 }); // clear any residual
  const capturePromise = r.capture({ rateHz: 50, durationSec: 4.0 });
  // Give the capture a tick to arm, then issue the step.
  await sleep(50);
  const movePromise = r.moveTo(
    { x: 1500, y: 0, yaw: 0 },
    { tolMm: 40, timeoutMs: 4000 },
  ).catch((e) => log.warn(`moveTo rejected: ${e}`));

  const [samples] = await Promise.all([capturePromise, movePromise]);
  log.record("step_samples", { n: samples.length, first: samples[0], last: samples[samples.length - 1] });
  log.info(`mpc_step_response captured ${samples.length} samples`);
};
