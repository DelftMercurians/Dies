// Hexagon dance formation
// 6 robots arranged in a hexagon, alternating positions in a wave pattern

globalThis.scenario = {
  name: "dance_hexagon",
  team: "blue",
  robots: [0, 1, 2, 3, 4, 5],
  env: "either",
};

globalThis.run = async function run({ team, world, log }) {
  const ROBOT_IDS = [0, 1, 2, 3, 4, 5];
  const CENTER_X = 0;
  const CENTER_Y = 0;
  const RADIUS = 1000; // mm radius of hexagon

  try {
    log.info("=== Hexagon Dance Scenario ===");
    log.info("6 robots in a hexagon, alternating positions");

    // Generate hexagon positions
    const hexagonPositions = ROBOT_IDS.map((id) => {
      const angle = (id / 6) * 2 * Math.PI;
      return {
        x: CENTER_X + RADIUS * Math.cos(angle),
        y: CENTER_Y + RADIUS * Math.sin(angle),
        yaw: 0,
      };
    });

    // Initial positioning
    const initialMovePromises = ROBOT_IDS.map((id) => {
      return team.robot(id).moveTo({
        x: hexagonPositions[id].x,
        y: hexagonPositions[id].y,
        yaw: hexagonPositions[id].yaw,
      });
    });

    await Promise.all(initialMovePromises);
    log.info("Robots in hexagon formation");

    // Track which robot is at which hexagon position
    let robotAtPosition = [0, 1, 2, 3, 4, 5]; // robot at position i

    // Alternating dance loop
    let iteration = 0;
    while (true) {
      // Swap odd robots: positions 1 -> 3 -> 5 -> 1
      // Swap even robots: positions 0 -> 2 -> 4 -> 0
      
      const oddPositions = [1, 3, 5];
      const evenPositions = [0, 2, 4];

      const movePromises = [];
      const newRobotAtPosition = [...robotAtPosition];

      if (iteration % 2 === 0) {
        // Odd positions rotate
        for (let i = 0; i < oddPositions.length; i++) {
          const currentPos = oddPositions[i];
          const nextPos = oddPositions[(i + 1) % oddPositions.length];
          const robotAtCurrentPos = robotAtPosition[currentPos];
          
          movePromises.push(
            team.robot(robotAtCurrentPos).moveTo({
              x: hexagonPositions[nextPos].x,
              y: hexagonPositions[nextPos].y,
              yaw: hexagonPositions[nextPos].yaw,
            })
          );
          newRobotAtPosition[nextPos] = robotAtCurrentPos;
        }
      } else {
        // Even positions rotate
        for (let i = 0; i < evenPositions.length; i++) {
          const currentPos = evenPositions[i];
          const nextPos = evenPositions[(i + 1) % evenPositions.length];
          const robotAtCurrentPos = robotAtPosition[currentPos];
          
          movePromises.push(
            team.robot(robotAtCurrentPos).moveTo({
              x: hexagonPositions[nextPos].x,
              y: hexagonPositions[nextPos].y,
              yaw: hexagonPositions[nextPos].yaw,
            })
          );
          newRobotAtPosition[nextPos] = robotAtCurrentPos;
        }
      }

      await Promise.all(movePromises);
      robotAtPosition = newRobotAtPosition;
      iteration++;
    }

  } catch (e) {
    log.error(`Error in hexagon dance scenario: ${e.message}`);
    throw e;
  }
};
