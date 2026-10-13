import { useMemo, useState } from "react";
import { Share2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  isVaultLink,
  openVaultLink,
  pushLogToVault,
  useLogs,
  useSendCommand,
  useVaultLogs,
} from "../api";
import { LogInfo, VaultLogInfo } from "../bindings";

function fmtTime(unix: number): string {
  return new Date(unix * 1000).toLocaleString();
}

function fmtDuration(s: number | undefined): string {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function fmtSize(b: number): string {
  return b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`;
}

function matches(needle: string, ...fields: (string | null | undefined)[]) {
  return fields.some((f) => (f ?? "").toLowerCase().includes(needle));
}

/**
 * Centered modal listing recorded logs with metadata + quick search. "Local"
 * lists this machine's logs (with a share-to-vault button); "Vault" lists the
 * shared log server. Pasting a vault link into the search box opens it.
 */
export function LogBrowserModal({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [tab, setTab] = useState("local");
  const logs = useLogs(open && tab === "local");
  const vault = useVaultLogs(open && tab === "vault");
  const sendCommand = useSendCommand();
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  const needle = query.toLowerCase();
  const pastedLink = isVaultLink(query) ? query.trim() : null;

  const filtered = useMemo(
    () =>
      logs.filter((l) =>
        matches(needle, l.name, l.blue_strategy, l.yellow_strategy)
      ),
    [logs, needle]
  );
  const filteredVault = useMemo(
    () =>
      vault.logs.filter((l) =>
        matches(
          needle,
          l.name,
          l.blue_strategy,
          l.yellow_strategy,
          l.blue_team_name,
          l.yellow_team_name
        )
      ),
    [vault.logs, needle]
  );

  const load = (l: LogInfo) => {
    sendCommand({ type: "LoadLog", data: { path: l.path } });
    onOpenChange(false);
  };

  const openLink = async (link: string) => {
    setBusy(link);
    const ok = await openVaultLink(link);
    setBusy(null);
    if (ok) onOpenChange(false);
  };

  const share = async (l: LogInfo) => {
    setBusy(l.path);
    await pushLogToVault(l.path);
    setBusy(null);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Load recorded log</DialogTitle>
        </DialogHeader>
        <Input
          autoFocus
          placeholder="Search by name or strategy, or paste a shared log link…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && pastedLink) openLink(pastedLink);
          }}
        />
        {pastedLink ? (
          <button
            onClick={() => openLink(pastedLink)}
            disabled={busy !== null}
            className="mt-2 block w-full rounded border border-accent-cyan px-2 py-2 text-left text-sm text-text-std hover:bg-bg-overlay disabled:opacity-50"
          >
            {busy === pastedLink ? "Fetching…" : "Open shared log"}{" "}
            <span className="font-mono text-[11px] text-text-dim">
              {pastedLink}
            </span>
          </button>
        ) : (
          <Tabs value={tab} onValueChange={setTab} className="mt-2">
            <TabsList>
              <TabsTrigger value="local">Local</TabsTrigger>
              <TabsTrigger value="vault">Vault</TabsTrigger>
            </TabsList>
            <TabsContent value="local">
              <div className="mt-2 max-h-96 divide-y divide-border-subtle overflow-y-auto">
                {filtered.length === 0 ? (
                  <Empty>No logs found</Empty>
                ) : (
                  filtered.map((l) => (
                    <div
                      key={l.path}
                      className="flex items-center hover:bg-bg-overlay"
                    >
                      <button
                        onClick={() => load(l)}
                        className="block min-w-0 flex-1 px-2 py-2 text-left"
                      >
                        <div className="font-mono text-sm text-text-std">
                          {l.name}
                          {l.is_zip ? " (zip)" : ""}
                        </div>
                        <div className="text-[11px] text-text-dim">
                          {fmtTime(l.session_start_unix)} ·{" "}
                          {fmtDuration(l.duration_s ?? undefined)} ·{" "}
                          {l.frame_count ?? "?"} frames ·{" "}
                          {l.is_simulation ? "sim" : "live"}
                          {l.blue_strategy ? ` · ${l.blue_strategy}` : ""}
                        </div>
                      </button>
                      <button
                        title="Upload to vault and copy share link"
                        onClick={() => share(l)}
                        disabled={busy !== null}
                        className="mr-2 rounded p-1.5 text-text-dim hover:bg-bg-elevated hover:text-text-bright disabled:opacity-40"
                      >
                        <Share2
                          className={`h-4 w-4 ${busy === l.path ? "animate-pulse" : ""}`}
                        />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </TabsContent>
            <TabsContent value="vault">
              <div className="mt-2 max-h-96 divide-y divide-border-subtle overflow-y-auto">
                {vault.error ? (
                  <Empty>Log vault unreachable: {vault.error}</Empty>
                ) : vault.isLoading ? (
                  <Empty>Loading…</Empty>
                ) : filteredVault.length === 0 ? (
                  <Empty>No logs in vault</Empty>
                ) : (
                  filteredVault.map((l) => (
                    <VaultRow
                      key={l.id}
                      log={l}
                      busy={busy === l.url}
                      disabled={busy !== null}
                      onOpen={() => openLink(l.url)}
                    />
                  ))
                )}
              </div>
              {vault.vaultUrl && (
                <div className="mt-1 text-right text-[10px] text-text-muted">
                  <a
                    href={vault.vaultUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="hover:text-text-dim"
                  >
                    {vault.vaultUrl}
                  </a>
                </div>
              )}
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="py-6 text-center text-sm text-text-dim">{children}</div>
  );
}

function VaultRow({
  log: l,
  busy,
  disabled,
  onOpen,
}: {
  log: VaultLogInfo;
  busy: boolean;
  disabled: boolean;
  onOpen: () => void;
}) {
  const kind = l.is_match ? "match" : l.is_simulation ? "sim" : "live";
  const sides = [
    l.blue_team_name ?? l.blue_strategy,
    l.yellow_team_name ?? l.yellow_strategy,
  ]
    .filter(Boolean)
    .join(" vs ");
  return (
    <button
      onClick={onOpen}
      disabled={disabled}
      className="block w-full px-2 py-2 text-left hover:bg-bg-overlay disabled:opacity-60"
    >
      <div className="font-mono text-sm text-text-std">
        {l.name}
        {busy ? " — fetching…" : ""}
      </div>
      <div className="text-[11px] text-text-dim">
        {l.session_start_unix != null ? fmtTime(l.session_start_unix) : "—"} ·{" "}
        {fmtDuration(l.duration_s ?? undefined)} · {l.frame_count ?? "?"} frames
        · {kind} · {fmtSize(l.size_stored)}
        {sides ? ` · ${sides}` : ""}
      </div>
    </button>
  );
}
