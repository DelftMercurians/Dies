// Lineup in the D formation as for delft scenario
// All 6 robots line up in a semicircle around (-3500, 0)

globalThis.scenario = {
  name: "lineup_D",
  team: "blue",
  robots: [0, 1, 2, 3, 4, 5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const ROBOT_IDS = [0, 1, 2, 3, 4, 5];
  const CENTER_X = -3300;
  const CENTER_Y = 0;
  const RADIUS = 1000; // mm
  
  try {
    log.info("=== Lineup D Scenario ===");
    log.info("Positioning all 6 robots in a semicircle around (-3500, 0)");

    // Position robots in a semicircle (right half, x > -3500)
    // Angles range from -90° to +90° (evenly spaced)
    const movePromises = ROBOT_IDS.map((id) => {
      const angleStep = Math.PI / (ROBOT_IDS.length - 1); // Divide 180° by (n-1)
      const angle = -Math.PI / 2 + id * angleStep; // Start at -90°
      
      const x = CENTER_X + RADIUS * Math.cos(angle);
      const y = CENTER_Y + RADIUS * Math.sin(angle);
      const yaw = angle + Math.PI / 2; // Point towards center
      
      return team.robot(id).moveTo({ x, y, yaw });
    });

    await Promise.all(movePromises);
    log.info("All robots positioned in semicircle");

  } catch (e) {
    log.error(`Error in lineup scenario: ${e.message}`);
    throw e;
  }
};
