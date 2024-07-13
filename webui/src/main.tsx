import "./index.css";

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { WorldDataProvider, startWsClient } from "./api";
import { Toaster } from "./components/ui/sonner";
import { TooltipProvider } from "./components/ui/tooltip";

startWsClient();

ReactDOM.createRoot(document.getElementById("app")!).render(
  <React.StrictMode>
    <TooltipProvider>
      <WorldDataProvider>
        <App />
        <Toaster />
      </WorldDataProvider>
    </TooltipProvider>
  </React.StrictMode>
);
