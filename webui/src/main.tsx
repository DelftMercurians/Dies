import "./index.css";

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { WorldFrameProvider, startWsClient } from "./api";
import { Toaster } from "./components/ui/sonner";
import { TooltipProvider } from "./components/ui/tooltip";

startWsClient();

ReactDOM.createRoot(document.getElementById("app")!).render(
  <React.StrictMode>
    <TooltipProvider>
      <WorldFrameProvider>
        <App />
        <Toaster />
      </WorldFrameProvider>
    </TooltipProvider>
  </React.StrictMode>
);
