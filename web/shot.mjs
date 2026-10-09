/* Screenshot helper: captures all major routes from the preview server. */
import { chromium } from "playwright";

const ROUTES = [
  ["overview", "/#/"],
  ["models", "/#/models"],
  ["model-detail", "/#/models/smollm-135m"],
  ["benchmarks", "/#/benchmarks"],
  ["benchmark-detail", "/#/benchmarks/cv-screening"],
  ["experiments", "/#/experiments"],
  ["runner", "/#/experiments/new"],
  ["experiment-detail", "/#/experiments/exp-2026-09-hiring-sweep"],
  ["compare", "/#/compare"],
  ["tasks", "/#/tasks"],
  ["task-detail", "/#/tasks/exp-2026-smollm-ster-000"],
  ["insights", "/#/insights"],
];

const base = process.argv[2] ?? "http://localhost:4173";
const outDir = process.argv[3] ?? ".qa/screenshots";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
for (const [name, route] of ROUTES) {
  await page.goto(base + route, { waitUntil: "networkidle" });
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: true });
  console.log("captured", name);
}
await browser.close();
