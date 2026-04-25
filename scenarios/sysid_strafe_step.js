// Strafe-axis step excitation + capture, useful for identifying the slower
// y-axis dynamics separately from forward.

globalThis.scenario = {
  name: "sysid_strafe_step",
  team: "blue",
  robots: [1],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const r = team.robot(5);
  // await world.addRobot({ team: "blue", id: 1, x: 0, y: -1000, yaw: 0 });
  await r.moveTo({ x: 0, y: -1000, yaw: 0 }, { tolMm: 60, timeoutMs: 5000 });
  await r.waitStopped({ timeoutMs: 2000 });

  const samples = await r.captureWhileExciting({
    excitation: step({
      axis: "strafe",
      magnitude: 50000,
      holdSec: 4.0,
      duration: 1.0,
    }),
    rateHz: 20,
    durationSec: 4.0,
  });
  log.record("samples", { samples });
  log.info(`strafe-step: captured ${samples.length} samples`);
};
