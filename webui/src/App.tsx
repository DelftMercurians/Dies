import Field from "./views/Field";

const App: React.FC = () => {
  return (
    <main className="w-full h-full flex flex-col bg-background">
      <div>
        <label>
          Select Scenario:
          {/* <select
            id="scenario-select"
            value={selectedScenario.type}
            onChange={handleScenarioChange}
          >
            {scenarios.map((scenario) => (
              <option key={scenario} value={scenario}>
                {scenario}
              </option>
            ))}
          </select> */}
        </label>
        <label>
          Simulation:
          {/* <button onClick={handleStartSimulation}>Start</button>
          <button onClick={handleStopSimulation}>Stop</button> */}
        </label>
      </div>

      <Field />

      <div>
        <h3>Controls</h3>
        <ul>
          <li>
            Use <strong>W,A,S,D</strong> to move the robot
          </li>
          <li>
            Use <strong>Q,E</strong> to rotate the robot
          </li>
          <li>
            Hold <strong>Space</strong> to use the dribbler
          </li>
          <li>
            Press <strong>V</strong> to kick (not implemented yet, should also
            allow charging the kick + showing this)
          </li>
        </ul>
      </div>
    </main>
  );
};

export default App;
