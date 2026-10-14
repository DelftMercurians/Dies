// Heart dance formation
// 6 robots form a heart shape, then rotate in place

globalThis.scenario = {
  name: "dance_heart",
  team: "blue",
  robots: [0, 1, 2, 3, 4, 5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const ROBOT_IDS = [0, 1, 2, 3, 4, 5];
  const CENTER_X = 0;
  const CENTER_Y = 0;

  try {
    log.info("=== Heart Dance Scenario ===");
    log.info("6 robots forming a heart shape, then rotating in place");

    // Heart shape positions
    const heartPositions = [
      { x: CENTER_X - 600, y: CENTER_Y + 800, id: 0 },   // Top left bump
      { x: CENTER_X + 600, y: CENTER_Y + 800, id: 1 },   // Top right bump
      { x: CENTER_X - 800, y: CENTER_Y + 200, id: 2 },   // Middle left
      { x: CENTER_X, y: CENTER_Y + 200, id: 3 },         // Middle center
      { x: CENTER_X + 800, y: CENTER_Y + 200, id: 4 },   // Middle right
      { x: CENTER_X, y: CENTER_Y - 600, id: 5 },         // Bottom point
    ];

    // Initial positioning
    const initialMovePromises = heartPositions.map((pos) => {
      return team.robot(pos.id).moveTo({
        x: pos.x,
        y: pos.y,
        yaw: 0,
      });
    });

    await Promise.all(initialMovePromises);
    log.info("Robots in heart formation");

    // Rotation on the spot loop
    let yaw = 0;
    const ROTATION_SPEED = 0.5; // radians per update
    while (true) {
      yaw = (yaw + ROTATION_SPEED) % (2 * Math.PI);

      const rotationPromises = heartPositions.map((pos) => {
        return team.robot(pos.id).moveTo({
          x: pos.x,
          y: pos.y,
          yaw: yaw,
        });
      });

      await Promise.all(rotationPromises);
    }

  } catch (e) {
    log.error(`Error in heart dance scenario: ${e.message}`);
    throw e;
  }
};
