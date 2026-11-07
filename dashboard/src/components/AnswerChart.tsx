import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Answer } from "../api";
import { fmt } from "../format";

// Validated categorical slots, in order (dataviz check on the dark surface). One series -> slot 1.
const SERIES = ["#3987E5", "#D95926", "#199E70", "#C98500"];
const AXIS = { fill: "#A3ACB6", fontSize: 11 };

type Chart = NonNullable<Answer["chart"]>;

export default function AnswerChart({ chart }: { chart: Chart }) {
  const rows = chart.rows.slice(0, 40).map((r) => Object.fromEntries(chart.columns.map((c, i) => [c, r[i]])));
  if (chart.type === "table" || !chart.y.length) {
    return (
      <div className="max-h-72 overflow-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr className="text-left text-faint">
              {chart.columns.map((c) => (
                <th key={c} className="sticky top-0 bg-panel py-1.5 pr-4 font-mono font-normal">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="tabular">
            {chart.rows.slice(0, 100).map((r, i) => (
              <tr key={i} className="border-t border-line">
                {r.map((v, j) => (
                  <td key={j} className="py-1.5 pr-4 text-muted">
                    {typeof v === "number" ? fmt(v, Number.isInteger(v) ? 0 : 1) : String(v ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const tooltip = (
    <Tooltip
      cursor={{ fill: "rgba(255,255,255,0.04)", stroke: "#2E343E" }}
      contentStyle={{ background: "#171A20", border: "1px solid #2E343E", borderRadius: 8, fontSize: 12 }}
      labelStyle={{ color: "#ECEEF1" }}
      itemStyle={{ color: "#A3ACB6" }}
      formatter={(v: number) => fmt(v, Number.isInteger(v) ? 0 : 1)}
    />
  );
  const legend =
    chart.y.length > 1 ? (
      <div className="flex gap-4 pb-1">
        {chart.y.map((y, i) => (
          <span key={y} className="flex items-center gap-1.5 text-xs text-muted">
            <span className="h-2 w-2 rounded-[2px]" style={{ background: SERIES[i % SERIES.length] }} />
            {y}
          </span>
        ))}
      </div>
    ) : null;

  if (chart.type === "line") {
    return (
      <figure className="m-0 flex flex-col gap-2">
        {chart.title && <figcaption className="text-xs text-muted">{chart.title}</figcaption>}
        {legend}
        <div className="h-64">
          <ResponsiveContainer>
            <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="#232830" vertical={false} />
              <XAxis dataKey={chart.x} tick={AXIS} stroke="#2E343E" />
              <YAxis tick={AXIS} stroke="#2E343E" width={56} tickFormatter={(v) => fmt(v)} />
              {tooltip}
              {chart.y.map((y, i) => (
                <Line key={y} type="monotone" dataKey={y} stroke={SERIES[i % SERIES.length]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </figure>
    );
  }

  const height = Math.min(420, 28 * rows.length + 40);
  return (
    <figure className="m-0 flex max-w-[760px] flex-col gap-2">
      {chart.title && <figcaption className="text-xs text-muted">{chart.title}</figcaption>}
      {legend}
      <div style={{ height }}>
        <ResponsiveContainer>
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 4 }} barCategoryGap={6}>
            <CartesianGrid stroke="#232830" horizontal={false} />
            <XAxis type="number" tick={AXIS} stroke="#2E343E" tickFormatter={(v) => fmt(v)} />
            <YAxis type="category" dataKey={chart.x} tick={AXIS} stroke="#2E343E" width={150} interval={0} />
            {tooltip}
            {chart.y.map((y, i) => (
              <Bar key={y} dataKey={y} fill={SERIES[i % SERIES.length]} radius={[0, 4, 4, 0]} maxBarSize={18} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
