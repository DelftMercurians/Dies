// JetBrains Mono font - mission control aesthetic
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";

import "./index.css";

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { PrimaryTeamProvider, TeamDataProvider, startWsClient } from "./api";
import { Toaster } from "./components/ui/sonner";
import { TooltipProvider } from "./components/ui/tooltip";

startWsClient();

ReactDOM.createRoot(document.getElementById("app")!).render(
  <React.StrictMode>
    <TooltipProvider>
      <TeamDataProvider>
        <PrimaryTeamProvider>
          <App />
          <Toaster />
        </PrimaryTeamProvider>
      </TeamDataProvider>
    </TooltipProvider>
  </React.StrictMode>
);
