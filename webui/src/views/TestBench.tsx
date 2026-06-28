import { FC, useEffect, useMemo, useState } from "react";
import {
  BenchMotionMode,
  BenchOneShot,
  PlayerFeedbackMsg,
  SysStatus,
  UiCommand,
} from "@/bindings";
import {
  useBenchTelemetry,
  useBenchKeyboardControl,
  useSendCommand,
  useStatus,
} from "@/api";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

// ---------------------------------------------------------------------------
// Command helpers
// ---------------------------------------------------------------------------

type Send = (cmd: UiCommand) => void;

const oneShot = (send: Send, robot_id: number, kind: BenchOneShot) =>
  send({ type: "Bench", data: { type: "OneShot", data: { robot_id, kind } } });

const broadcast = (send: Send, kind: BenchOneShot) =>
  send({ type: "Bench", data: { type: "Broadcast", data: { kind } } });

// Set (kind set) or clear (kind undefined) a continuously-held command.
const setHoldCmd = (
  send: Send,
  robot_id: number,
  kind: BenchOneShot | undefined
) =>
  send({ type: "Bench", data: { type: "SetHold", data: { robot_id, kind } } });

const stopRobot = (send: Send, robot_id: number) =>
  send({ type: "Bench", data: { type: "Stop", data: { robot_id } } });

const stopAll = (send: Send) =>
  send({ type: "Bench", data: { type: "StopAll" } });

const takeControl = (send: Send, robot_id: number, taken: boolean) =>
  send({
    type: "Bench",
    data: { type: "TakeControl", data: { robot_id, taken } },
  });

const sendSetChannel = (
  send: Send,
  robot_id: number | undefined,
  channel: number
) =>
  send({
    type: "Bench",
    data: { type: "SetChannel", data: { robot_id, channel } },
  });

// ---------------------------------------------------------------------------
// Telemetry formatting helpers
// ---------------------------------------------------------------------------

const ROBOT_CMD_LABELS: Record<number, string> = {
  0: "NONE",
  1: "ARM",
  2: "DISARM",
  3: "DISCHARGE",
  4: "KICK",
  5: "ARM_COUNTER_KICK",
  6: "ARM_TIMED_KICK",
  7: "ARM_REFLEX_KICK",
  10: "HEADING_CONTROL",
  11: "YAW_RATE_CONTROL",
  12: "CHIP",
  16: "COAST",
  32: "BEEP",
  48: "REBOOT",
  64: "POWER_BOARD_OFF",
  80: "CALIBRATE_IMU",
  96: "CALIBRATE_BB",
};

const statusColor = (status?: SysStatus): string => {
  switch (status) {
    case SysStatus.Ok:
    case SysStatus.Ready:
    case SysStatus.Safe:
    case SysStatus.Standby:
      return "text-accent-green";
    case SysStatus.Emergency:
    case SysStatus.Overtemp:
    case SysStatus.NoReply:
      return "text-accent-red";
    case SysStatus.Armed:
    case SysStatus.Stop:
    case SysStatus.Cooldown:
      return "text-accent-amber";
    case undefined:
      return "text-text-muted";
    default:
      return "text-accent-cyan";
  }
};

const fmt = (v: number | undefined, digits = 1, unit = ""): string =>
  v === undefined || v === null ? "—" : `${v.toFixed(digits)}${unit}`;

const Stat: FC<{ label: string; children: React.ReactNode }> = ({
  label,
  children,
}) => (
  <div className="flex justify-between gap-2 text-sm">
    <span className="text-text-dim">{label}</span>
    <span className="font-mono text-text-std">{children}</span>
  </div>
);

const StatusDot: FC<{ status?: SysStatus; label: string }> = ({
  status,
  label,
}) => (
  <div className="flex items-center gap-1.5 text-sm">
    <span className={`${statusColor(status)}`}>●</span>
    <span className="text-text-dim">{label}</span>
    <span className={`${statusColor(status)} font-mono`}>
      {status ?? "—"}
    </span>
  </div>
);

const VoltageBar: FC<{ label: string; value?: number; max: number }> = ({
  label,
  value,
  max,
}) => {
  const pct = value ? Math.min(100, (value / max) * 100) : 0;
  const color =
    value === undefined
      ? "bg-text-muted"
      : value < max * 0.72
        ? "bg-accent-red"
        : value < max * 0.8
          ? "bg-accent-amber"
          : "bg-accent-green";
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-4 text-text-dim">{label}</span>
      <div className="relative h-3 flex-1 bg-bg-overlay">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-14 text-right font-mono text-text-std">
        {fmt(value, 1, "V")}
      </span>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Robot grid card
// ---------------------------------------------------------------------------

const RobotCard: FC<{
  player: PlayerFeedbackMsg;
  selected: boolean;
  disabled: boolean;
  onSelect: () => void;
  send: Send;
}> = ({ player, selected, disabled, onSelect, send }) => (
  <button
    onClick={onSelect}
    className={`flex flex-col gap-1 border p-2 text-left transition-colors ${
      selected
        ? "border-accent-cyan bg-bg-overlay"
        : "border-border-subtle bg-bg-surface hover:bg-bg-overlay"
    }`}
  >
    <div className="flex items-center justify-between">
      <span className="font-semibold text-text-bright">Robot {player.id}</span>
      <span className={statusColor(player.primary_status)}>
        {player.primary_status ?? "—"}
      </span>
    </div>
    <VoltageBar label="L" value={player.pack_voltages?.[0]} max={14} />
    <VoltageBar label="R" value={player.pack_voltages?.[1]} max={14} />
    <div className="flex justify-between text-sm text-text-dim">
      <span>Cap {fmt(player.kicker_cap_voltage, 0, "V")}</span>
      <span className={player.breakbeam_ball_detected ? "text-accent-amber" : ""}>
        {player.breakbeam_ball_detected ? "BALL" : "no ball"}
      </span>
    </div>
    <div className="mt-1 flex gap-1">
      <Button
        variant="destructive"
        size="xs"
        disabled={disabled}
        onClick={(e) => {
          e.stopPropagation();
          stopRobot(send, player.id);
        }}
      >
        Stop
      </Button>
      <Button
        variant="outline"
        size="xs"
        disabled={disabled}
        onClick={(e) => {
          e.stopPropagation();
          oneShot(send, player.id, { type: "Beep" });
        }}
      >
        Beep
      </Button>
    </div>
  </button>
);

// ---------------------------------------------------------------------------
// Single-robot focus panel (full command surface)
// ---------------------------------------------------------------------------

const MOTOR_LABELS = ["Motor 0", "Motor 1", "Motor 2", "Motor 3", "Dribbler"];

const RobotFocus: FC<{
  player: PlayerFeedbackMsg;
  disabled: boolean;
  send: Send;
}> = ({ player, disabled, send }) => {
  const [mode, setMode] = useState<BenchMotionMode>(BenchMotionMode.Local);
  const [speed, setSpeed] = useState(0.5); // m/s
  const [angularSpeed, setAngularSpeed] = useState(90); // deg/s
  const [dribble, setDribble] = useState(0); // raw dribbler value
  const [kickTime, setKickTime] = useState(2500); // raw kick_time_i [ms]
  const [driving, setDriving] = useState(false);
  const [channel, setChannel] = useState("");
  const [heading, setHeading] = useState("");
  // Command currently being held (continuously re-sent), or null.
  const [hold, setHoldState] = useState<"Arm" | "ArmReflex" | null>(null);

  const robotId = player.id;

  // Toggle a continuously-held command; selecting a different one replaces it.
  const toggleHold = (kind: "Arm" | "ArmReflex") => {
    const next = hold === kind ? null : kind;
    setHoldState(next);
    setHoldCmd(send, robotId, next ? { type: next } : undefined);
  };

  // Clear any hold (local + backend), e.g. on Disarm.
  const clearHold = () => {
    if (hold !== null) {
      setHoldState(null);
      setHoldCmd(send, robotId, undefined);
    }
  };

  // Keyboard driving streams SetMotion at 30 Hz while taken.
  useBenchKeyboardControl({
    robotId: driving && !disabled ? robotId : null,
    mode,
    speed,
    angularSpeedDegPerSec: angularSpeed,
    dribbleSpeed: dribble,
  });

  // Release control (and any hold) when the focused robot changes or the panel
  // unmounts. `takeControl(false)` clears the backend hold; reset local state.
  useEffect(() => {
    setHoldState(null);
    return () => {
      takeControl(send, robotId, false);
      setHoldCmd(send, robotId, undefined);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [robotId]);

  // Disabling (e.g. a live executor starts) drops driving and any hold.
  useEffect(() => {
    if (disabled) {
      if (driving) {
        setDriving(false);
        takeControl(send, robotId, false);
      }
      clearHold();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disabled]);

  const toggleDriving = (val: boolean) => {
    setDriving(val);
    takeControl(send, robotId, val);
  };

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-1">
      <h3 className="text-text-bright">
        Robot {robotId}
        {player.firmware_version && (
          <span className="ml-2 font-mono text-sm text-text-dim">
            fw {player.firmware_version.major}.{player.firmware_version.minor}.
            {player.firmware_version.patch}
          </span>
        )}
      </h3>

      {/* Telemetry */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 border border-border-subtle p-2">
        <StatusDot status={player.primary_status} label="Main" />
        <StatusDot status={player.kicker_status} label="Kicker" />
        <StatusDot status={player.imu_status} label="IMU" />
        <StatusDot status={player.tof_status} label="TOF" />
        <Stat label="Cap V">{fmt(player.kicker_cap_voltage, 1, "V")}</Stat>
        <Stat label="Main I">{fmt(player.main_board_current, 1, "A")}</Stat>
        <Stat label="Reflex">{player.reflex_kick_state ?? "—"}</Stat>
        <Stat label="Kick cnt">{player.smart_kick_counter ?? "—"}</Stat>
        <Stat label="Breakbeam">
          {player.breakbeam_sensor_ok === false
            ? "n/i"
            : player.breakbeam_ball_detected
              ? "BALL"
              : "no ball"}
        </Stat>
        <Stat label="BB raw">{player.breakbeam_raw ?? "—"}</Stat>
        <Stat label="TOF">
          {player.tof_xy ? `${player.tof_xy[0]},${player.tof_xy[1]}` : "—"}
        </Stat>
        <Stat label="Loop µs">
          {player.avg_loop_time_us ?? "—"}/{player.max_loop_time_us ?? "—"}
        </Stat>
        <Stat label="IMU θz">
          {fmt(player.imu_readings?.[2], 2, " rad")}
        </Stat>
        <Stat label="IMU ωz">
          {fmt(player.imu_readings?.[5], 2, " rad/s")}
        </Stat>
      </div>

      {/* Motors */}
      <div className="border border-border-subtle p-2">
        <div className="mb-1 text-sm font-semibold text-text-dim">Motors</div>
        {MOTOR_LABELS.map((label, i) => (
          <div key={i} className="flex justify-between text-sm">
            <span className={statusColor(player.motor_statuses?.[i])}>
              {label}
            </span>
            <span className="font-mono text-text-std">
              {fmt(player.motor_speeds?.[i], 0, " rad/s")} ·{" "}
              {fmt(player.motor_currents?.[i], 1, "A")} ·{" "}
              {fmt(player.motor_temps?.[i], 0, "°C")}
            </span>
          </div>
        ))}
      </div>

      {/* Last command echo */}
      {player.last_command && (
        <div className="border border-border-subtle p-2 text-sm">
          <div className="mb-1 font-semibold text-text-dim">Last command</div>
          <div className="font-mono text-text-std">
            {ROBOT_CMD_LABELS[player.last_command.robot_command] ??
              player.last_command.robot_command}{" "}
            · vx {fmt(player.last_command.global_speed_x, 2)} · vy{" "}
            {fmt(player.last_command.global_speed_y, 2)} · ψ{" "}
            {fmt(player.last_command.heading_setpoint, 2)} · drib{" "}
            {player.last_command.dribbler_speed}
          </div>
        </div>
      )}

      {/* Motion controls */}
      <fieldset disabled={disabled} className="border border-border-subtle p-2">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm font-semibold text-text-dim">Drive</span>
          <div className="flex gap-1">
            {[BenchMotionMode.Local, BenchMotionMode.Global].map((m) => (
              <Button
                key={m}
                variant={mode === m ? "primary" : "outline"}
                size="xs"
                onClick={() => setMode(m)}
              >
                {m}
              </Button>
            ))}
          </div>
        </div>
        <SliderRow
          label="Speed"
          value={speed}
          min={0}
          max={3}
          step={0.05}
          unit=" m/s"
          onChange={setSpeed}
        />
        <SliderRow
          label="Yaw rate"
          value={angularSpeed}
          min={0}
          max={360}
          step={5}
          unit=" °/s"
          onChange={setAngularSpeed}
        />
        <SliderRow
          label="Dribbler"
          value={dribble}
          min={0}
          max={1000}
          step={10}
          unit=""
          onChange={setDribble}
        />
        <div className="mt-2 flex items-center justify-between">
          <label className="flex items-center gap-2 text-sm text-text-std">
            <Switch checked={driving} onCheckedChange={toggleDriving} />
            Take control (keyboard)
          </label>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => stopRobot(send, robotId)}
          >
            Stop
          </Button>
        </div>
        {driving && (
          <div className="mt-1 text-sm text-text-muted">
            WASD move · Q/E rotate · Space dribble
          </div>
        )}
      </fieldset>

      {/* Kicker */}
      <fieldset disabled={disabled} className="border border-border-subtle p-2">
        <div className="mb-2 text-sm font-semibold text-text-dim">Kicker</div>
        <SliderRow
          label="Kick time"
          value={kickTime}
          min={0}
          max={5000}
          step={50}
          unit=" ms"
          onChange={setKickTime}
        />
        <div className="mt-2 flex flex-wrap gap-1">
          <Button
            size="sm"
            variant={hold === "Arm" ? "primary" : "outline"}
            onClick={() => toggleHold("Arm")}
          >
            {hold === "Arm" ? "Arming…" : "Arm (hold)"}
          </Button>
          <Button
            size="sm"
            variant={hold === "ArmReflex" ? "primary" : "outline"}
            onClick={() => toggleHold("ArmReflex")}
          >
            {hold === "ArmReflex" ? "Arming reflex…" : "Arm reflex (hold)"}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              clearHold();
              oneShot(send, robotId, { type: "Disarm" });
            }}
          >
            Disarm
          </Button>
          <Button size="sm" variant="outline" onClick={() => oneShot(send, robotId, { type: "Discharge" })}>
            Discharge
          </Button>
          <Button
            size="sm"
            variant="primary"
            onClick={() => oneShot(send, robotId, { type: "Kick", data: { kick_time: kickTime } })}
          >
            Kick
          </Button>
        </div>
      </fieldset>

      {/* System + calibration */}
      <fieldset disabled={disabled} className="border border-border-subtle p-2">
        <div className="mb-2 text-sm font-semibold text-text-dim">
          System & calibration
        </div>
        <div className="flex flex-wrap gap-1">
          <Button size="sm" variant="outline" onClick={() => oneShot(send, robotId, { type: "Beep" })}>
            Beep
          </Button>
          <Button size="sm" variant="outline" onClick={() => oneShot(send, robotId, { type: "Coast" })}>
            Coast
          </Button>
          <Button size="sm" variant="outline" onClick={() => oneShot(send, robotId, { type: "CalibrateImu" })}>
            Cal IMU
          </Button>
          <Button size="sm" variant="outline" onClick={() => oneShot(send, robotId, { type: "CalibrateBreakbeam" })}>
            Cal BB
          </Button>
          <Button size="sm" variant="outline" onClick={() => oneShot(send, robotId, { type: "GetVersion" })}>
            Get version
          </Button>
          <Button size="sm" variant="outline" onClick={() => oneShot(send, robotId, { type: "ZeroHeading" })}>
            Zero heading
          </Button>
          <Button size="sm" variant="destructive" onClick={() => oneShot(send, robotId, { type: "Reboot" })}>
            Reboot
          </Button>
          <Button size="sm" variant="destructive" onClick={() => oneShot(send, robotId, { type: "Shutdown" })}>
            Shutdown
          </Button>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Input
            className="w-24"
            placeholder="heading"
            value={heading}
            onChange={(e) => setHeading(e.target.value)}
          />
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              const a = Number(heading);
              if (!Number.isNaN(a))
                oneShot(send, robotId, { type: "SetHeading", data: { angle: a } });
            }}
          >
            Set heading
          </Button>
          <Input
            className="w-24"
            placeholder="channel"
            value={channel}
            onChange={(e) => setChannel(e.target.value)}
          />
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              const c = Number(channel);
              if (!Number.isNaN(c)) sendSetChannel(send, robotId, c);
            }}
          >
            Set robot channel
          </Button>
        </div>
      </fieldset>
    </div>
  );
};

const SliderRow: FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}> = ({ label, value, min, max, step, unit, onChange }) => (
  <div className="flex items-center gap-3 py-0.5">
    <span className="w-20 text-sm text-text-dim">{label}</span>
    <Slider
      className="flex-1"
      value={[value]}
      min={min}
      max={max}
      step={step}
      onValueChange={([v]) => onChange(v)}
    />
    <span className="w-20 text-right font-mono text-sm text-text-std">
      {value}
      {unit}
    </span>
  </div>
);

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

const TestBenchModal: FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
}> = ({ open, onOpenChange }) => {
  const { data } = useBenchTelemetry(open);
  const { data: status } = useStatus();
  const send = useSendCommand();
  const [selected, setSelected] = useState<number | null>(null);
  const [baseChannel, setBaseChannel] = useState("");

  const blocked =
    status?.ui_mode === "Live" && status?.executor?.type === "RunningExecutor";

  // Aggregate all robots (bench addresses by raw id; robots usually land in
  // unknown_team without an id map). Dedupe by id.
  const robots = useMemo(() => {
    if (!data) return [] as PlayerFeedbackMsg[];
    const byId = new Map<number, PlayerFeedbackMsg>();
    for (const p of [...data.blue_team, ...data.yellow_team, ...data.unknown_team]) {
      if (!byId.has(p.id)) byId.set(p.id, p);
    }
    return [...byId.values()].sort((a, b) => a.id - b.id);
  }, [data]);

  const selectedPlayer = robots.find((r) => r.id === selected) ?? null;
  const base = data?.base_info;

  const handleOpenChange = (next: boolean) => {
    if (!next) stopAll(send); // safety: halt everything on close
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="flex h-[92vh] w-[95vw] max-w-[95vw] flex-col bg-bg-base">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-4">
            <span>Robot Test Bench</span>
            {base ? (
              <span className="flex items-center gap-3 text-sm font-normal text-text-dim">
                <span className={base.protocol_ok ? "text-accent-green" : "text-accent-red"}>
                  ● base {base.version}
                </span>
                <span>proto {base.protocol_version}</span>
                <span>{base.channel_mhz} MHz</span>
                <span>
                  radios{" "}
                  {base.radios_online.map((on, i) => (
                    <span key={i} className={on ? "text-accent-green" : "text-accent-red"}>
                      ●
                    </span>
                  ))}
                </span>
              </span>
            ) : (
              <span className="text-sm font-normal text-accent-red">
                ● basestation not connected
              </span>
            )}
          </DialogTitle>
        </DialogHeader>

        {blocked && (
          <div className="border border-accent-amber bg-accent-amber/10 px-3 py-2 text-sm text-accent-amber">
            A live executor is running — bench sending is disabled. Stop the
            executor to control robots. Telemetry stays live.
          </div>
        )}

        {/* Global command bar */}
        <fieldset disabled={blocked} className="flex flex-wrap items-center gap-2 border-b border-border-subtle pb-2">
          <Button variant="outline" size="sm" onClick={() => broadcast(send, { type: "Beep" })}>
            Beep all
          </Button>
          <Button variant="destructive" size="sm" onClick={() => broadcast(send, { type: "Reboot" })}>
            Reboot all
          </Button>
          <Button variant="destructive" size="sm" onClick={() => broadcast(send, { type: "Shutdown" })}>
            Shutdown all
          </Button>
          <Button variant="destructive" size="sm" onClick={() => stopAll(send)}>
            STOP ALL
          </Button>
          <div className="ml-auto flex items-center gap-2">
            <Input
              className="w-24"
              placeholder="base ch"
              value={baseChannel}
              onChange={(e) => setBaseChannel(e.target.value)}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const c = Number(baseChannel);
                if (!Number.isNaN(c)) sendSetChannel(send, undefined, c);
              }}
            >
              Set base channel
            </Button>
          </div>
        </fieldset>

        {/* Body: grid + focus */}
        <div className="flex min-h-0 flex-1 gap-4">
          <div className="w-[360px] shrink-0 overflow-y-auto">
            {robots.length > 0 ? (
              <div className="grid grid-cols-2 gap-2">
                {robots.map((p) => (
                  <RobotCard
                    key={p.id}
                    player={p}
                    selected={p.id === selected}
                    disabled={blocked}
                    onSelect={() => setSelected(p.id)}
                    send={send}
                  />
                ))}
              </div>
            ) : (
              <div className="p-4 text-sm text-text-muted">
                No robots online.
              </div>
            )}
          </div>
          <div className="min-w-0 flex-1 border-l border-border-subtle pl-4">
            {selectedPlayer ? (
              <RobotFocus
                key={selectedPlayer.id}
                player={selectedPlayer}
                disabled={blocked}
                send={send}
              />
            ) : (
              <div className="p-4 text-sm text-text-muted">
                Select a robot to control it.
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TestBenchModal;
