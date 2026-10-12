/** Built-in tutorial notebooks. Each one adds one idea to the last. */

export interface NotebookExample {
  id: string;
  title: string;
  cells: string[];
}

const MOVE = `// Drive the robot to the center
robot().set({ position: [0, 0] })
`;

const WAIT = `const R = robot()
R.set({ position: [1000, 0] })

// Wait until it is close
await until(() => v2.dist(R.state.position, [1000, 0]) < 30)
show("arrived")
`;

const HEADING = `const R = robot()

// yaw is the direction the robot faces
R.set({ position: [0, 0], yaw: deg(90) })
await wait(2)
R.set({ position: [0, 0], yaw: deg(-90) })
`;

const VELOCITY = `const R = robot()
R.set({ position: [-1000, 0] })
await wait(3)

// Drive with a speed instead of a target (mm/s)
R.set({ velocity: { Global: [500, 0] } })
await wait(2)
R.set({ velocity: { Global: [0, 0] } })
`;

const LOCAL = `const R = robot()
R.set({ position: [0, 0], yaw: deg(45) })
await wait(3)

// Local = relative to the robot. [500, 0] is forward.
R.set({ velocity: { Local: [500, 0] } })
await wait(1.5)
R.set({ velocity: { Local: [0, 0] } })
`;

const LOOP = `const R = robot()
R.set({ position: [-1000, 0] })
await wait(3)

// Run code every frame. Slow down near x = 1000.
for (;;) {
  await frame()
  const left = 1000 - R.state.position[0]
  if (left < 20) break
  R.set({ velocity: { Global: [clamp(2 * left, 100, 1000), 0] } })
}
R.set({ velocity: { Global: [0, 0] } })
`;

const LOG = `const R = robot()
R.set({ position: [0, 0] })
await wait(3)

// log() saves values in the log file
for (let i = 0; i < 60; i++) {
  await frame()
  R.log({ speed: v2.norm(R.state.velocity) })
  R.set({ velocity: { Global: [800, 0] } })
}
R.set({ velocity: { Global: [0, 0] } })
`;

const TRIAL = `const R = robot()
R.set({ position: [-1000, 0] })
await wait(3)

// A trial records every frame
const t = await trial("go", async () => {
  R.set({ position: [1000, 0] })
  await wait(3)
})

show(\`took \${t.duration.toFixed(2)} s\`)
show(t.rows(R).slice(0, 5))
`;

const PLOT = `const R = robot()
R.set({ position: [-1000, 0] })
await wait(3)

const t = await trial("go", async () => {
  R.set({ position: [1000, 0] })
  await wait(3)
})

// Plot a column against time
show(Plot.lineY(t.rows(R), { x: "t", y: "velocity_x" }))
`;

const PARAMS = `const R = robot()
const trials = []

// Repeat with different settings
for (const limit of [500, 1000, 2000]) {
  R.set({ position: [-1000, 0] })
  await wait(3)
  trials.push(await trial("go", { limit }, async () => {
    R.set({ position: [1000, 0], acceleration_limit: limit })
    await wait(3)
  }))
}

const rows = trials.flatMap((t) => t.rows(R))
show(Plot.plot({
  color: { legend: true },
  marks: [Plot.line(rows, { x: "t", y: "velocity_x", stroke: "limit" })],
}))
`;

const STATS = `// Numbers from each trial. Run after "Compare settings".
show(trials.map((t) => ({
  limit: t.params.limit,
  time: t.duration,
  top_speed: stats.maxAbs(t.col(R, "velocity_x")),
})))
`;

const SKILL = `const R = robot()

// Skills are the same moves strategies use.
// await waits until the skill is done.
const result = await R.skill({
  GoToPos: { position: [0, 1000], heading: deg(180), hold_ground: false, avoid_robots: true },
})
show(result)
`;

const STOP = `const R = robot()
R.set({ velocity: { Global: [300, 0] } })

// The Stop button (or Esc) stops every robot.
// release() gives the robot back.
await wait(1)
R.release()
`;

export const NOTEBOOK_EXAMPLES: NotebookExample[] = [
  { id: "move", title: "1. Move", cells: [MOVE] },
  { id: "wait", title: "2. Wait", cells: [WAIT] },
  { id: "heading", title: "3. Heading", cells: [HEADING] },
  { id: "velocity", title: "4. Speed", cells: [VELOCITY] },
  { id: "local", title: "5. Robot frame", cells: [LOCAL] },
  { id: "loop", title: "6. Every frame", cells: [LOOP] },
  { id: "log", title: "7. Log values", cells: [LOG] },
  { id: "trial", title: "8. Record a trial", cells: [TRIAL] },
  { id: "plot", title: "9. Plot", cells: [PLOT] },
  { id: "params", title: "10. Compare settings", cells: [PARAMS, STATS] },
  { id: "skill", title: "11. Skills", cells: [SKILL] },
  { id: "stop", title: "12. Stop and release", cells: [STOP] },
];
