import { useAtom } from "jotai";
import { Wrench } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { benchOpenAtom } from "@/api";
import TestBenchModal from "@/views/TestBench";

/**
 * Toolbar entry that opens the Robot Test Bench — a vision-free, direct-to-
 * basestation robot exerciser. Available any time, including with the executor
 * stopped.
 */
export default function TestBenchButton() {
  const [open, setOpen] = useAtom(benchOpenAtom);
  return (
    <>
      <SimpleTooltip title="Robot test bench (drive robots without vision)">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setOpen(true)}
          className="text-text-dim hover:text-text-std"
        >
          <Wrench className="h-4 w-4" />
        </Button>
      </SimpleTooltip>
      <TestBenchModal open={open} onOpenChange={setOpen} />
    </>
  );
}
