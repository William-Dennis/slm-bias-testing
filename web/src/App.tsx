import { HashRouter, Route, Routes } from "react-router-dom";
import { Shell } from "./components/shell";
import { BenchmarkDetail } from "./routes/BenchmarkDetail";
import { Benchmarks } from "./routes/Benchmarks";
import { Compare } from "./routes/Compare";
import { ExperimentDetail } from "./routes/ExperimentDetail";
import { Experiments } from "./routes/Experiments";
import { Insights } from "./routes/Insights";
import { ModelDetail } from "./routes/ModelDetail";
import { Models } from "./routes/Models";
import { Overview } from "./routes/Overview";
import { Runner } from "./routes/Runner";
import { TaskDetail } from "./routes/TaskDetail";
import { Tasks } from "./routes/Tasks";

export default function App() {
  return (
    <HashRouter>
      <Shell>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/models" element={<Models />} />
          <Route path="/models/:id" element={<ModelDetail />} />
          <Route path="/benchmarks" element={<Benchmarks />} />
          <Route path="/benchmarks/:id" element={<BenchmarkDetail />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/experiments/new" element={<Runner />} />
          <Route path="/experiments/:id" element={<ExperimentDetail />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/tasks/:id" element={<TaskDetail />} />
          <Route path="/insights" element={<Insights />} />
          <Route path="*" element={<Overview />} />
        </Routes>
      </Shell>
    </HashRouter>
  );
}
