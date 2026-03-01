import React from "react";
import { X, AlertTriangle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScriptError } from "@/bindings";

interface ScriptErrorDialogProps {
  error: ScriptError | null;
  open: boolean;
  onClose: () => void;
}

export const ScriptErrorDialog: React.FC<ScriptErrorDialogProps> = ({
  error,
  open,
  onClose,
}) => {
  if (!error) return null;

  const isSyntaxError = error.type === "Syntax";

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <DialogTitle>
              {isSyntaxError ? "Script Syntax Error" : "Script Runtime Error"}
            </DialogTitle>
          </div>
        </DialogHeader>

        <div className="space-y-4">
          {isSyntaxError && error.type === "Syntax" && (
            <>
              <div>
                <label className="text-sm font-medium text-text-dim">
                  Script File:
                </label>
                <p className="font-mono bg-bg-elevated p-2">
                  {error.data.script_path}
                </p>
              </div>

              {(error.data.line || error.data.column) && (
                <div>
                  <label className="text-sm font-medium text-text-dim">
                    Location:
                  </label>
                  <p>
                    {error.data.line && `Line ${error.data.line}`}
                    {error.data.line && error.data.column && ", "}
                    {error.data.column && `Column ${error.data.column}`}
                  </p>
                </div>
              )}

              <div>
                <label className="text-sm font-medium text-text-dim">
                  Error Message:
                </label>
                <div className="bg-red-900/20 border border-red-500/30 p-3 font-mono whitespace-pre-wrap">
                  {error.data.message}
                </div>
              </div>
            </>
          )}

          {!isSyntaxError && error.type === "Runtime" && (
            <>
              <div>
                <label className="text-sm font-medium text-text-dim">
                  Script File:
                </label>
                <p className="font-mono bg-bg-elevated p-2">
                  {error.data.script_path}
                </p>
              </div>

              <div>
                <label className="text-sm font-medium text-text-dim">
                  Function:
                </label>
                <p className="font-mono bg-bg-elevated p-2">
                  {error.data.function_name}
                </p>
              </div>

              <div>
                <label className="text-sm font-medium text-text-dim">
                  Team & Player:
                </label>
                <p>
                  Team: {error.data.team_color}
                  {error.data.player_id && `, Player: ${error.data.player_id}`}
                </p>
              </div>

              <div>
                <label className="text-sm font-medium text-text-dim">
                  Error Message:
                </label>
                <div className="bg-red-900/20 border border-red-500/30 p-3 font-mono whitespace-pre-wrap">
                  {error.data.message}
                </div>
              </div>
            </>
          )}
        </div>

        <div className="flex justify-end gap-2 pt-4">
          <Button onClick={onClose} variant="outline">
            <X className="w-4 h-4 mr-2" />
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
