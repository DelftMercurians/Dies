import "./index.css";

import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App.js";
import { WorldDataProvider } from "./api";
import { Toaster } from "./components/ui/sonner";

const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById("app")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <WorldDataProvider>
        <App />
        <Toaster />
      </WorldDataProvider>
    </QueryClientProvider>
  </React.StrictMode>
);
