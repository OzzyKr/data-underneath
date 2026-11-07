import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { useLogs } from "./api";
import { lastLog } from "./format";
import Shell from "./components/Shell";
import CsvPage from "./pages/CsvPage";
import ImagePage from "./pages/ImagePage";
import InsightsPage from "./pages/InsightsPage";

function Home() {
  const { data } = useLogs();
  if (!data) return null;
  const remembered = lastLog();
  const id = data.find((l) => l.log_id === remembered)?.log_id ?? data.find((l) => l.kind === "colour")?.log_id ?? data[0]?.log_id;
  return id ? <Navigate to={`/csv/${id}`} replace /> : <p className="p-12 text-muted">No logs yet. Run `lith dash-build`.</p>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<Home />} />
          <Route path="/csv/:logId" element={<CsvPage />} />
          <Route path="/image/:logId" element={<ImagePage />} />
          <Route path="/insights" element={<InsightsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
