Dies Framework

Yohan Branch

## Requirements

- CLI: Main entrypoint for the framework, allows for running with or without a UI
- UI Server: Serves the UI and communicates with the Executor and other APIs
  - Svelte Frontend: UI for the framework
- Executor: Responsible for running a match or a scenario, spinning up all required components, and running the selected strategies either with real robots or in a simulation
- World: The world model, which is used by the executor and the strategies to keep track of the state of the game. Includes the tracking of the ball, robots, and field markings.
- Strat Engine: Responsible for running the strategies, receives data from the World and sends commands back to the Executor
  - Strat API: The Python API for the strategies to use to interact with the world and send commands to the robots
- Tournament Env: The environment for and IRL match or scenario. Sets up communication with the robots and the vision system, and provides the World with data from the real world.
  - Base Station
  - Vision Client
  - Ref Client
- ER-Sim Env: The environment for a simulated match or scenario. Sets up communication with the simulated robots and the simulated vision system, and provides the World with data from the simulation.
  - ER-Sim
  - ER Autoref
  - Game Controller
- Scenario Mode: The mode for running a scenario, which is a single play of a match with a specific starting state. This is used for testing strategies.
  - Scenario Spec API: The Python API for defining a scenario
- Playback Mode: The mode for playing back a match from a log file.
- Logging: Our custom logging system, which logs data from the World, the srtategies, including debug information, and other core components.
- Python API: The Python API for the framework, which can be used to set up for example a simulated match from a Jupyter notebook.
- Gym: A Gym environment for the framework, which can be used to train strategies using reinforcement learning.
- Log tools: Tools for analyzing and visualizing log files.

## Development Plan

1. Design the critical interfaces between the components:
   1. Strat API: Write a simple example strategy to figure out the best API design
   2. World: Figure out what data the World needs to keep track of at a minimum
   3. Executor: Figure out what commands the strategies need to be able to send to the Executor, and what other data needs to pass through the Executor
   4. Robot commands: Figure out what data the robots need and in what format
2. Implement the most important components:
   1. World
   2. Executor
   3. Strat Engine
   4. Logging
   5. CLI
3. Design the UI and implement it, along with the UI Server on top of the Executor API
4. Create a wrapper for ER-Sim, the autoref, and the game controller, and implement the ER-Sim Env
5. Implement the Base Station, Vision/Ref Client, and the Tournament Env
6. Design and implement the Scenario Spec API
7. Implement the Playback Mode
8. Implement the Python API on top of the Executor API
9. Implement the Gym environment and the log tools
