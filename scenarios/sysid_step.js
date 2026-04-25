// Strafe-axis step excitation + capture, useful for identifying the slower
// y-axis dynamics separately from forward.

globalThis.scenario = {
  name: "sysid_step",
  team: "blue",
  robots: [4],
  env: "either",
};

globalThis.run = async function run({ team, world, log, sysid }) {
  const r = team.robot(4);
  // await world.addRobot({ team: "blue", id: 1, x: 0, y: -1000, yaw: 0 });
  await r.moveTo({ x: -2500, y: 0, yaw: 0 }, { tolMm: 60 });
  await r.waitStopped({ timeoutMs: 1000 });

  const samples = await r.captureWhileExciting({
    excitation: step({
      axis: "forward",
      magnitude: 5000,
      holdSec: 2.0,
    }),
    rateHz: 10,
    durationSec: 10.0,
  });
  log.record("samples", { samples });

  const fit = await sysid.fit(samples);
  log.record("fit", fit);
  log.info(
    `strafe-step fit: converged=${fit.converged} rms=[${fit.residualRms.join(", ")}]`,
  );
};
