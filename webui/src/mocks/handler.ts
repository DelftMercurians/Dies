import { rest } from "msw";
import type { ServerState } from "../types";

export const handlers = [
  rest.get("/api/state", (req, res, ctx) => {
    const serverState: ServerState = {
      world: {
        ball: {
          position: { x: 0, y: 0 },
          velocity: { x: 0, y: 0 },
        },
        own_players: [
          {
            position: { x: -400, y: 0 },
            velocity: { x: 0, y: 0 },
          },
        ],
        opp_players: [
          {
            position: { x: 400, y: 0 },
            velocity: { x: 0, y: 0 },
          },
        ],
        field_dimensions: { x: 10400, y: 7400 },
      },
      player_states: [
        {
          role: "goalie",
          target: { x: 0, y: 0 },
        },
      ],
    };
    return res(ctx.status(200), ctx.json(serverState));
  }),
];
