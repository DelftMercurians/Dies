# API Reference Overview

This section provides a detailed reference for all the functions and nodes available in the Dies Rhai scripting environment.

The API can be categorized as follows:

- **Behavior Nodes**: These are the building blocks of your behavior tree. They control the flow of execution.
  - **Composite Nodes**: `Select`, `Sequence`, `ScoringSelect`.
  - **Decorator Nodes**: `Guard`, `Semaphore`.
- **Skills (Action Nodes)**: These are the leaf nodes of the tree that perform actual actions, like moving or kicking.
- **Helpers**: Utility functions, for example for creating vectors or working with player ID's.
- **The Situation Object**: An object passed to condition and scorer functions, providing access to the current world state.

Each part of the API is detailed in the subsequent pages. When writing scripts, you will be combining these components to create complex and intelligent behaviors for the robots.
