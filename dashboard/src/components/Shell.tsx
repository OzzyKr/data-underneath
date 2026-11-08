import { useState } from "react";
import { NavLink, Outlet, useLocation, useParams } from "react-router-dom";
import { ChartLine, Image as ImageIcon, ScanSearch, Table2, Upload } from "lucide-react";
import { useLogs, useMeta } from "../api";
import { lastLog } from "../format";
import LogPicker from "./LogPicker";
import ProviderSwitch from "./ProviderSwitch";
import RunDrawer from "./RunDrawer";

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <rect x="2" y="4" width="18" height="3" rx="1.5" fill="#8F74EA" />
      <rect x="2" y="9.5" width="18" height="3" rx="1.5" fill="#56606D" />
      <rect x="2" y="15" width="18" height="3" rx="1.5" fill="#56606D" />
    </svg>
  );
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm no-underline transition-colors ${
    isActive ? "bg-panel-2 font-medium text-ink" : "text-muted hover:bg-hover hover:text-ink"
  }`;

export default function Shell() {
  const { logId } = useParams();
  const { pathname } = useLocation();
  const { data: logs } = useLogs();
  const { data: meta } = useMeta();
  const [drawer, setDrawer] = useState(false);
  const current = logId ?? lastLog() ?? logs?.[0]?.log_id ?? "";
  const onInsights = pathname.startsWith("/insights");
  const onLayout = pathname.startsWith("/layout");

  return (
    <div className="flex min-h-full flex-wrap bg-bg">
      <nav aria-label="Main" className="flex shrink-0 grow basis-52 flex-col gap-9 border-r border-line bg-nav px-4 py-7 md:max-w-52">
        <div className="flex items-center gap-2.5 px-2.5">
          <Logo />
          <span className="text-sm font-semibold tracking-tight">Data Underneath</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <NavLink to={current ? `/csv/${current}` : "/"} className={navClass}>
            <Table2 size={16} strokeWidth={1.75} aria-hidden="true" />
            CSV
          </NavLink>
          <NavLink to={current ? `/image/${current}` : "/"} className={navClass}>
            <ImageIcon size={16} strokeWidth={1.75} aria-hidden="true" />
            Image
          </NavLink>
          <NavLink to="/insights" className={navClass}>
            <ChartLine size={16} strokeWidth={1.75} aria-hidden="true" />
            Insights
          </NavLink>
          <NavLink to="/layout" className={navClass}>
            <ScanSearch size={16} strokeWidth={1.75} aria-hidden="true" />
            Layout
          </NavLink>
        </div>
        <p className="mt-auto px-3 text-xs leading-relaxed text-faint">
          {meta ? `${meta.logs} logs` : ""}
          {meta?.holdout_hidden ? ` · ${meta.holdout_hidden} holdout hidden` : ""}
        </p>
      </nav>

      <div className="flex min-w-0 shrink grow-[999] basis-[560px] flex-col">
        <header className="flex flex-wrap items-center gap-3 border-b border-line px-6 py-4 lg:px-12">
          {onLayout ? (
            <span className="text-sm text-muted">Layout detector · step 1</span>
          ) : onInsights ? (
            <span className="text-sm text-muted">All logs</span>
          ) : (
            <LogPicker current={logId} />
          )}
          <div className="ml-auto flex flex-wrap items-center gap-2.5">
            {onInsights && <ProviderSwitch />}
            <button
              type="button"
              onClick={() => setDrawer(true)}
              className="flex h-10 items-center gap-2 rounded-lg border border-line-2 px-3.5 text-sm text-ink transition-colors hover:bg-hover"
            >
              <Upload size={15} strokeWidth={1.75} aria-hidden="true" />
              Run a scan
            </button>
          </div>
        </header>
        <main className="w-full min-w-0 flex-1">
          <Outlet />
        </main>
      </div>
      {drawer && <RunDrawer onClose={() => setDrawer(false)} />}
    </div>
  );
}
