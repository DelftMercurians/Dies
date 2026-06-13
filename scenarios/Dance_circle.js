// Lineup in a circle formation around a center point
// 6 robots arranged in a circle at the center of the field

globalThis.scenario = {
  name: "dance_circle",
  team: "blue",
  robots: [0, 1, 2, 3, 4, 5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const ROBOT_IDS = [0, 1, 2, 3, 4, 5];
  const CENTER_X = -2000; // Middle of field
  const CENTER_Y = 0; // Middle of field
  const RADIUS = 1000; // mm radius of circle
  const ROTATION_SPEED = 0.2; // radians per second
  let startTime = Date.now();

  try {
    log.info("=== Dance Circle Scenario ===");
    log.info("Positioning 6 robots in a rotating circle formation around center");

    // Initial positions
    const initialPositions = ROBOT_IDS.map((id) => {
      const angle = (id / ROBOT_IDS.length) * 2 * Math.PI;
      const x = CENTER_X + RADIUS * Math.cos(angle);
      const y = CENTER_Y + RADIUS * Math.sin(angle);
      const yaw = Math.atan2(CENTER_Y - y, CENTER_X - x);

      return { x, y, yaw, id };
    });

    const initialMovePromises = initialPositions.map((pos) => {
      return team.robot(pos.id).moveTo({ x: pos.x, y: pos.y, yaw: pos.yaw });
    });

    await Promise.all(initialMovePromises);
    log.info("Robots in initial circle formation, starting rotation");

    // Continuous rotation loop
    while (true) {
      const elapsedSeconds = (Date.now() - startTime) / 1000;
      const baseAngle = elapsedSeconds * ROTATION_SPEED;

      const positions = ROBOT_IDS.map((id) => {
        const angle = (id / ROBOT_IDS.length) * 2 * Math.PI + baseAngle;
        const x = CENTER_X + RADIUS * Math.cos(angle);
        const y = CENTER_Y + RADIUS * Math.sin(angle);
        
        // Calculate yaw to face the center
        const yaw = Math.atan2(CENTER_Y - y, CENTER_X - x);

        return { x, y, yaw, id };
      });

      const movePromises = positions.map((pos) => {
        return team.robot(pos.id).moveTo({ x: pos.x, y: pos.y, yaw: pos.yaw });
      });

      await Promise.all(movePromises);
    }

  } catch (e) {
    log.error(`Error in dance circle scenario: ${e.message}`);
    throw e;
  }
};
