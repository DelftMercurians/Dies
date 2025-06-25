# Introduction to Behavior Trees

A Behavior Tree (BT) is a mathematical model of plan execution used in computer science, robotics, control systems and video games. They are a way of describing the "brain" of an AI-controlled character or agent.

## Core Concepts

A BT is a tree of nodes that controls the flow of decision-making. Each node, when "ticked" (or executed), returns a status:

- `Success`: The node has completed its task successfully.
- `Failure`: The node has failed to complete its task.
- `Running`: The node is still working on its task and needs more time.

There are several types of nodes:

### 1. Action Nodes (or Leaf Nodes)

These are the leaves of the tree and represent actual actions the agent can perform. In our system, these are called **Skills**, like `Kick`, `FetchBall`, or `GoToPosition`. An Action Node will typically return `Running` while the action is in progress, and `Success` or `Failure` upon completion.

### 2. Composite Nodes

These nodes have one or more children and control how their children are executed. The most common types are:

- **Sequence**: Executes its children one by one in order. It returns `Failure` as soon as one of its children fails. If a child returns `Running`, the Sequence node also returns `Running` and will resume from that child on the next tick. It returns `Success` only if all children succeed. A Sequence is like a logical **AND**.

- **Select** (or Fallback): Executes its children one by one in order. It returns `Success` as soon as one of its children succeeds. If a child returns `Running`, the Select node also returns `Running` and will resume from that child on the next tick. It returns `Failure` only if all children fail. A Select is like a logical **OR**.

- **ScoringSelect**: A more advanced version of `Select`. It evaluates a score for each child and picks the one with the highest score to execute. It includes hysteresis to prevent rapid switching between behaviors.

### 3. Decorator Nodes

These nodes have a single child and modify its behavior or its return status. Common examples include:

- **Guard** (or Condition): Checks a condition. If the condition is true, it ticks its child node and returns the child's status. If the condition is false, it returns `Failure` without executing the child.

- **Semaphore**: Limits the number of agents that can run a certain part of the tree simultaneously. This is crucial for team coordination, preventing all robots from trying to do the same thing at once (e.g., everyone chasing the ball).

## Behavior Trees in Dies

In the Dies framework, a behavior tree is constructed for each robot on every tick (if it doesn't have one already). This tree dictates the robot's actions for that tick. The tree is built and executed based on the current state of the game, which is provided to the BT as a `RobotSituation` object. This object contains all the information a robot needs to make decisions, such as ball position, player positions, and game state.
