// Drive robot #1 around a square using moveTo waypoints (direct path → MTP).
// Verifies: MTP position tracking, manual_override wiring, scenario lifecycle.

globalThis.scenario = {
  name: "manual_drive",
  team: "blue",
  robots: [5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const r = team.robot(0);

  // Drop the robot in sim at a known starting pose.
  // await world.addRobot({ team: "blue", id: 4, x: 0, y: 0, yaw: 0 });

  const HOME = { x: -1500, y: 0, yaw: 0 };
  const side = 1000; // mm
  const waypoints = [
    { x: HOME.x - side, y: HOME.y - side, yaw: Math.PI },
    { x: HOME.x - side, y: HOME.y + side, yaw: Math.PI },
    { x: HOME.x + side, y: HOME.y + side, yaw: Math.PI },
    { x: HOME.x + side, y: HOME.y - side, yaw: Math.PI },
  ];

  r.startRecording({ rateHz: 10 });
  for (const wp of waypoints) {
    log.info(`heading to (${wp.x}, ${wp.y})`);
    await r.moveTo(wp, { tolMm: 80 });
  }
  r.stop();
  const samples = r.stopRecording();
  log.dumpCsv("square", samples);

  log.info("manual_drive done");
};
