// Multi-axis excitation sweep: forward + strafe, several passes each with
// alternating sign and a chirp tail. Re-centers the robot between passes so
// nothing runs off the field. Concatenates all samples for a single fit.

globalThis.scenario = {
  name: "sysid_multi_axis",
  team: "blue",
  robots: [4],
  env: "either",
};

const HOME = { x: -1500, y: 0, yaw: 0 };

globalThis.run = async function run({ team, world, log, sysid }) {
  const r = team.robot(4);

  const recenter = async () => {
    await r.moveTo(HOME, { tolMm: 60, timeoutMs: 6000 });
    await r.waitStopped({ thresholdMmPerSec: 40, timeoutMs: 2000 });
  };

  // Each pass: { axis, kind: "step"|"chirp", params, durationSec }.
  // Sign is encoded in step magnitude; chirp covers both directions implicitly.
  const passes = [
    { axis: "forward", kind: "step", mag: 2000, durationSec: 2.0 },
    { axis: "forward", kind: "step", mag: -2000, durationSec: 2.0 },
    { axis: "forward", kind: "step", mag: 4000, durationSec: 2.0 },
    // { axis: "forward", kind: "chirp", f0: 0.3, f1: 2.5, amp: 500, durationSec: 8.0 },

    { axis: "strafe", kind: "step", mag: 2000, durationSec: 2.0 },
    { axis: "strafe", kind: "step", mag: -2000, durationSec: 2.0 },
    { axis: "strafe", kind: "step", mag: 4000, durationSec: 2.0 },
    // { axis: "strafe", kind: "chirp", f0: 0.3, f1: 2.5, amp: 500, durationSec: 8.0 },
  ];

  const allSamples = [];

  for (let i = 0; i < passes.length; i++) {
    const p = passes[i];
    log.info(`pass ${i + 1}/${passes.length}: ${p.axis} ${p.kind}`);

    await recenter();

    const excitation = p.kind === "step"
      ? step({
        axis: p.axis,
        magnitude: p.mag,
        holdSec: p.durationSec / 2,
        duration: p.durationSec,
      })
      : chirp({
        axis: p.axis,
        f0: p.f0,
        f1: p.f1,
        amp: p.amp,
        duration: p.durationSec,
      });

    const samples = await r.captureWhileExciting({
      excitation,
      rateHz: 10,
      durationSec: p.durationSec,
    });
    log.record(`samples_${i}_${p.axis}_${p.kind}`, { samples });
    allSamples.push(...samples);
  }

  r.stop();

  log.info(`captured ${allSamples.length} samples across ${passes.length} passes`);
  log.dumpCsv("mulit_axis", allSamples);
  const fit = await sysid.fit(allSamples);
  log.record("fit", fit);
  log.info(
    `multi-axis fit: converged=${fit.converged} iters=${fit.iters} rms=[${fit.residualRms.join(", ")}]`,
  );
};
