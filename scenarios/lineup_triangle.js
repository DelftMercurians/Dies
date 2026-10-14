// Lineup in a triangle formation towards the middle
// 6 robots arranged in a triangle (1-2-3 rows) at the center of the field

globalThis.scenario = {
  name: "lineup_triangle",
  team: "blue",
  robots: [0, 1, 2, 3, 4, 5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const ROBOT_IDS = [0, 1, 2, 3, 4, 5];
  const CENTER_X = -2000; // Middle of field
  const CENTER_Y = 0; // Middle of field
  const SPACING = 800; // mm between robots

  try {
    log.info("=== Lineup Triangle Scenario ===");
    log.info("Positioning 6 robots in a play button triangle formation");

    // Triangle formation pointing right (play button shape)
    const positions = [
      // Left point
      { x: CENTER_X - SPACING*0.8, y: CENTER_Y, id: 0 },
      // Upper middle
      { x: CENTER_X - SPACING*0.8, y: CENTER_Y + SPACING, id: 1 },
      // Lower middle
      { x: CENTER_X - SPACING*0.8, y: CENTER_Y - SPACING, id: 2 },
      // Upper right
      { x: CENTER_X + SPACING*0.8, y: CENTER_Y, id: 3 },
      // Middle right
      { x: CENTER_X, y: CENTER_Y + SPACING * 0.5, id: 4 },
      // Lower right
      { x: CENTER_X, y: CENTER_Y - SPACING * 0.5, id: 5 },
    ];

    const movePromises = positions.map((pos) => {
      return team.robot(pos.id).moveTo({ x: pos.x, y: pos.y, yaw: 0 });
    });

    await Promise.all(movePromises);
    log.info("All robots positioned in triangle formation");

  } catch (e) {
    log.error(`Error in lineup triangle scenario: ${e.message}`);
    throw e;
  }
};
