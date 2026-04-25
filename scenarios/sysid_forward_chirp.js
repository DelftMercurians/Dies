// Open-loop forward-axis chirp with synchronized capture → offline LM fit.
// Verifies: excitation generator, captureWhileExciting, sysid.fit.

globalThis.scenario = {
  name: "sysid_forward_chirp",
  team: "blue",
  robots: [1],
  env: "either",
};

globalThis.run = async function run({ team, world, log, sysid }) {
  const r = team.robot(5);
  // await world.addRobot({ team: "blue", id: 1, x: -1500, y: 0, yaw: 0 });
  await r.moveTo({ x: -1500, y: 0, yaw: 0 }, { tolMm: 60, timeoutMs: 5000 });
  await r.waitStopped({ thresholdMmPerSec: 40, timeoutMs: 2000 });

  log.info("starting chirp capture");
  const samples = await r.captureWhileExciting({
    excitation: chirp({
      axis: "forward",
      f0: 0.3,
      f1: 2.5,
      amp: 500,
      duration: 8.0,
    }),
    rateHz: 20,
    durationSec: 8.0,
  });
  log.info(`captured ${samples.length} samples`);

  const fit = await sysid.fit(samples);
  log.record("fit", fit);
  log.info(
    `fit iters=${fit.iters} converged=${fit.converged} rms=[${fit.residualRms.join(", ")}]`,
  );
};
