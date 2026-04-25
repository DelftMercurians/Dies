// Drive robot #1 around a square using moveTo waypoints (direct path → MTP).
// Verifies: MTP position tracking, manual_override wiring, scenario lifecycle.

globalThis.scenario = {
  name: "manual_drive",
  team: "blue",
  robots: [1],
  env: "Real",
};

globalThis.run = async function run({ team, world, log }) {
  const r = team.robot(4);

  // Drop the robot in sim at a known starting pose.
  // await world.addRobot({ team: "blue", id: 4, x: 0, y: 0, yaw: 0 });

  const side = 1000; // mm
  const waypoints = [
    { x: side, y: 0, yaw: 0 },
    { x: side, y: side, yaw: Math.PI / 2 },
    { x: 0, y: side, yaw: Math.PI },
    { x: 0, y: 0, yaw: -Math.PI / 2 },
  ];

  for (const wp of waypoints) {
    log.info(`heading to (${wp.x}, ${wp.y})`);
    await r.moveTo(wp, { tolMm: 80, timeoutMs: 10000 });
  }

  r.stop();
  log.info("manual_drive done");
};
