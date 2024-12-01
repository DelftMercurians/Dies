<!-- - InteractiveMainLoop
  - UiMessage
  - UiUpdate
  - Simulator
  - SslClient
    - VisionMsg
    - RefereeMsg
  - BsClient
    - RobotCmd
    - RobotFeedback
  - Tracker
    - PlayerTracker
      - PlayerModel
    - BallTracker
      - BallModel
    - GameStateTracker
    - World (TrackerData)
      - Player
      - Ball
      - Field
      - GameState
  - TeamMap
    - Team
      - TeamController
        - PlayerController
          - RobotController
          - Skills
            - GoTo
            - Face
            - FetchBall
            - OrbitBall
            - Pass
            - Receive
        - TeamCmd
      - StrategyServer
- StrategyClient
  - StrategyApi
- TestMainLoop
- PlaybackMainLoop -->

## Framework Component Relationships

<details>
    <summary>Legend</summary>
    <ul>
        <li>A `*--` B: Ownership (A can only exist as part of B)</li>
        <li>A `o..` B: Aggregation (A owns B, but B can exist independently)</li>
        <li>A `<..` B: Dependency (A does not store B, but makes use of it)</li>
    </ul>
</details>

```mermaid
classDiagram
    %%  Cli is the main entry point
    Cli o.. WebUi
    Cli o.. TestMainLoop

    %% WebUi owns the main loops
    WebUi o.. InteractiveMainLoop
    WebUi o.. PlaybackMainLoop

    %% Main components owned by TestMainLoop
    TestMainLoop <.. Simulator
    TestMainLoop <.. TeamMap

    %% Main components owned by InteractiveMainLoop
    InteractiveMainLoop <.. Simulator
    InteractiveMainLoop <.. SslClient
    InteractiveMainLoop <.. BsClient
    InteractiveMainLoop <.. TeamMap

    PlaybackMainLoop *-- Logplayer

    %% TeamMap owns Teams
    TeamMap *-- Team

    %% Team owns core components
    Team *-- Tracker
    Team *-- TeamController
    Team *-- StrategyServer

    %% Tracker hierarchy
    Tracker *-- PlayerTracker
    Tracker *-- BallTracker
    Tracker *-- GameStateTracker

    %% Controller hierarchy
    TeamController *-- PlayerController
    PlayerController *-- RobotController
    PlayerController *-- Skills

    %% Skills collection
    Skills o-- GoTo
    Skills o-- Face
    Skills o-- FetchBall
    Skills o-- OrbitBall
    Skills o-- Pass
    Skills o-- Receive

    %% Strategy client/server relationship
    StrategyServer ..> StrategyApi : IPC
    StrategyApi <-- StrategyClient

    class Cli {
        main()
    }

    class Skills {
        <<collection>>
    }
    class StrategyServer {
        <<IPC Server>>
    }
    class StrategyApi {

    }
```

## Data Flow

**Interactive Mode (simulation or live)**

```mermaid
flowchart TB
    %% Define all components first
    UI[Web UI]
    WS[Web Server]
    ML[Main Loop]
    TR[Tracker]
    CT[Controller]
    SS[Strategy Server]
    SA[Strategy API]
    ST[Strategy Process]
    SIM[Simulator]
    SSL[SSL/BS Client]
    TM[Team]
    CLI[CLI]

    %% Group the UI layer
    subgraph UserInterface[User Interface]
        UI
        WS
        CLI
    end

    %% Group the core system components
    subgraph CoreSystem[Core]
        ML
        SIM
        SSL
    end

    %% Group the team components
    subgraph TeamSystem[Team]
        TM
        TR
        CT
        SS
    end

    %% Group the strategy components
    subgraph StrategySystem[Strategy]
        SA
        ST
    end

    %% Now define all the connections
    CLI -->|Config| WS
    ML -->|Debug data & Status| WS
    WS -->|Player Overrides| TM
    WS <-->|Start/Stop| ML
    UI <-->|UiMessages & UiData| WS
    ML -->|Simulator Cmds| SIM

    SSL -->|Vision/GC Updates| TR
    SIM -->|Vision/GC Updates| TR

    TR -->|Tracker Data| TM
    TM -->|Tracker Data| CT
    TM -->|Tracker Data| SS

    SS -->|Team Cmds| CT
    SS <--->|Team Cmds & Tracker Data| SA
    SA <-->|IPC| ST

    SSL <-->|Robot Cmds| CT

    TR -->|World Frame| WS
```

**Test Mode (automated)**

```mermaid
flowchart TB
    %% Main components - note the reduced set
    TML[Test Main Loop]
    TR[Tracker]
    CT[Controller]
    SS[Strategy Server]
    SA[Strategy API]
    ST[Strategy Process]
    SIM[Simulator]
    TM[Team]

    %% Test configuration and control
    TML -->|Test Scenario Commands| SIM

    %% Simulated vision flow
    SIM -->|Simulated Vision/GC Updates| TR

    %% Tracker data flow - through team
    TR -->|Tracker Data| TM
    TM -->|Tracker Data| CT
    TM -->|Tracker Data| SS

    %% Strategy flow remains the same
    SS -->|Team Cmds| CT
    SS <-->|Team Cmds & Tracker Data| SA
    SA <-->|IPC| ST

    %% Robot control now goes directly to simulator
    CT -->|Robot Commands| SIM

    %% Add subgraph for test components
    subgraph TestFramework[Test Framework]
        TML
    end

    %% Add subgraph for team components
    subgraph Team
        TM
        TR
        CT
        SS
    end
```
