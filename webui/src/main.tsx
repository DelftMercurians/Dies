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
