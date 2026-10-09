/* Interaction verification: runner flow, compare deep-link, CmdK. */
import { chromium } from "playwright";

const base = process.argv[2] ?? "http://localhost:4173";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

// 1. Runner end-to-end
await page.goto(base + "/#/experiments/new", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
console.log("steps:", await page.locator(".step-pill").count());
for (let i = 0; i < 2; i++) {
  const n = await page.locator("button", { hasText: "Continue" }).count();
  console.log(`iter ${i}: continue buttons = ${n}`);
  await page.locator("button", { hasText: "Continue" }).first().click();
  await page.waitForTimeout(400);
}
await page.locator("button", { hasText: "Review" }).click();
await page.waitForTimeout(400);
await page.locator("button", { hasText: "Run experiment" }).last().click();
await page.waitForTimeout(2500);
await page.screenshot({ path: ".qa/screenshots/runner-running.png" });
await page.waitForURL(/exp-user-/, { timeout: 15000 });
console.log("runner OK ->", page.url());
await page.waitForTimeout(800);

// 2. Compare deep link from Insights
await page.goto(base + "/#/compare?models=qwen25-05b,qwen25-15b,qwen3-06b", { waitUntil: "networkidle" });
await page.waitForTimeout(500);
const cards = await page.locator(".stat-card .stat-label").allTextContents();
console.log("compare cards:", cards.join(" | "));

// 3. Tasks deep link with flag
await page.goto(base + "/#/tasks?flag=disagreement", { waitUntil: "networkidle" });
await page.waitForTimeout(500);
const flagVal = await page.locator("select").nth(3).inputValue();
console.log("tasks flag preset:", flagVal);

// 4. CmdK
await page.goto(base + "/#/", { waitUntil: "networkidle" });
await page.keyboard.press("Meta+k");
await page.waitForTimeout(400);
await page.keyboard.type("nimble");
await page.waitForTimeout(300);
await page.screenshot({ path: ".qa/screenshots/cmdk.png" });
const items = await page.locator(".cmdk-item").count();
console.log("cmdk items for 'nimble':", items);

await browser.close();
