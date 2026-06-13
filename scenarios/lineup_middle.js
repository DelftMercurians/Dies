// Lineup in middle scenario
// All 6 robots line up in the middle of the field

globalThis.scenario = {
  name: "lineup_middle",
  team: "blue",
  robots: [0, 1, 2, 3, 4, 5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const ROBOT_IDS = [0, 1, 2, 3, 4, 5];
  const SPACING = 300; // 300mm spacing between robots
  const FIELD_WIDTH = 6000; 
  
  try {
    log.info("=== Lineup Middle Scenario ===");
    log.info("Positioning all 6 robots in a line at the middle of the field");

    // Position robots horizontally in the middle (y=0), spread along x-axis
    const movePromises = ROBOT_IDS.map((id) => {
      const xOffset = (id - 2.5) * SPACING; // Center the lineup
      return team.robot(id).moveTo({ x: xOffset - FIELD_WIDTH / 3, y: 0, yaw: Math.PI/2 });
    });

    await Promise.all(movePromises);
    log.info("All robots positioned in line at middle of field");

  } catch (e) {
    log.error(`Error in lineup scenario: ${e.message}`);
    throw e;
  }
};
