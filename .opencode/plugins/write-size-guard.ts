/**
 * Blocks oversized write tool calls before they reach the editor.
 *
 * Large generated writes can fail or corrupt files in some agent runtimes. Keep
 * this guard active by default; split large file creation into smaller edits.
 */
import { appendFile, mkdir } from "fs/promises"
import { dirname, join, resolve } from "path"
import { randomUUID } from "crypto"

const WARN_LINES = 100
const BLOCK_LINES = 150

async function logViolation(
  directory: string,
  sessionId: string,
  filePath: string,
  lineCount: number,
  blocked: boolean,
) {
  try {
    const out = join(
      resolve(directory),
      ".opencode",
      "metrics",
      "write-size-violations.jsonl",
    )
    await mkdir(dirname(out), { recursive: true })
    await appendFile(
      out,
      JSON.stringify({
        ts: new Date().toISOString(),
        session_id: sessionId,
        file_path: filePath || "",
        line_count: lineCount,
        blocked,
      }) + "\n",
      "utf-8",
    )
  } catch {
    // Observability must not break the guard.
  }
}

export const WriteSizeGuard = async ({ directory }: { directory: string }) => {
  const sessionId = randomUUID()

  return {
    "tool.execute.before": async (input?: any, output?: any) => {
      if (input?.tool !== "write") return

      const args = input?.args || output?.args || {}
      const content = typeof args.content === "string" ? args.content : ""
      const filePath = args.filePath || ""
      const lineCount = content.split("\n").length

      if (lineCount > BLOCK_LINES) {
        await logViolation(directory, sessionId, filePath, lineCount, true)
        throw new Error(
          `Write blocked: ${lineCount} lines exceeds the ${BLOCK_LINES}-line limit. ` +
            "Create a small placeholder file, then append content in smaller edits.",
        )
      }

      if (lineCount > WARN_LINES) {
        await logViolation(directory, sessionId, filePath, lineCount, false)
        console.error(
          `[write-size-guard] Warning: ${filePath} write has ${lineCount} lines.`,
        )
      }
    },
  }
}
