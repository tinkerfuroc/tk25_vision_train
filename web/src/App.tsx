import { useQuery } from "@tanstack/react-query";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { api } from "./api/rest";
import { ClipsPage } from "./pages/Clips";
import { DatasetsPage } from "./pages/Datasets";
import { LabelPage } from "./pages/Label";
import { TestPage } from "./pages/Test";

export function App() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <header className="border-b border-slate-800 px-6 py-3 flex items-center justify-between">
          <Link to="/" className="text-lg font-semibold tracking-wide hover:text-emerald-300">
            tk_vision
          </Link>
          <div className="text-xs text-slate-400 flex gap-4">
            <Link to="/datasets" className="hover:text-slate-200">Datasets</Link>
            <span>v{health.data?.version ?? "…"}</span>
            <span>
              sam3:{" "}
              <span
                className={
                  health.data?.sam3_loaded ? "text-emerald-400" : "text-amber-400"
                }
              >
                {health.data?.sam3_loaded ? "loaded" : "off"}
              </span>
            </span>
            {health.data?.gpu ? (
              <span>
                {health.data.gpu}/{health.data.dtype}
              </span>
            ) : null}
          </div>
        </header>
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<ClipsPage />} />
            <Route path="/datasets" element={<DatasetsPage />} />
            <Route path="/label/:clipId" element={<LabelPage />} />
            <Route path="/test/:runId/:clipId" element={<TestPage />} />
            <Route path="/test/:runId" element={<TestPage />} />
            <Route path="/test" element={<TestPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
