/**
 * pi verifier-agent extension: RL reward commands + read-only transcript access.
 *
 * Registers two slash commands the verifier agent uses after reviewing a
 * finished multi-agent task:
 *
 *   /rl/set_reward <session_id> <reward>   -> POST gateway /rl/set_reward
 *   /export_trajectories <session_id>      -> POST gateway /export_trajectories
 *
 * and a read-only ``read_task_messages`` tool that fetches the Multica
 * ``task_message`` transcript the verifier judges from. The verifier sets
 * rewards BEFORE the training loop harvests via /export_trajectories, honoring
 * the reward-before-export ordering (export is terminal -- it revokes the
 * session).
 *
 * Config (no secrets hardcoded):
 *   AREAL_GATEWAY_URL, AREAL_ADMIN_API_KEY   -- the inference gateway
 *   MULTICA_BASE_URL, MULTICA_API_KEY        -- read-only transcript access
 *
 * Placement: copy this directory to the verifier agent's ``.pi/extensions/``.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import {
  exportTrajectories,
  gatewayConfigFromEnv,
  setReward,
  type FetchLike,
} from "./rl_gateway.ts";

/** Parse ``<session_id> <reward>`` from a command argument string. */
export function parseSetRewardArgs(
  args: string,
): { sessionId: string; reward: number } {
  const parts = args.trim().split(/\s+/).filter(Boolean);
  if (parts.length !== 2) {
    throw new Error("usage: /rl/set_reward <session_id> <reward>");
  }
  const reward = Number(parts[1]);
  if (!Number.isFinite(reward)) {
    throw new Error(`reward must be a number, got ${parts[1]}`);
  }
  return { sessionId: parts[0]!, reward };
}

/** Build the read-only Multica task_message transcript URL. */
export function buildTaskMessagesUrl(
  multicaBaseUrl: string,
  taskId: string,
): string {
  if (!taskId) throw new Error("read_task_messages requires a task_id");
  return `${multicaBaseUrl.replace(/\/+$/, "")}/api/tasks/${encodeURIComponent(taskId)}/messages`;
}

export default function verifierRlExtension(pi: ExtensionAPI): void {
  const fetchImpl = fetch as unknown as FetchLike;

  pi.registerCommand("rl/set_reward", {
    description: "Assign an RL reward to one agent session (verifier credit).",
    handler: async (args: string, ctx) => {
      try {
        const { sessionId, reward } = parseSetRewardArgs(args);
        await setReward(fetchImpl, gatewayConfigFromEnv(), { sessionId, reward });
        ctx.ui.notify(`set_reward session=${sessionId} reward=${reward}`, "info");
      } catch (err) {
        ctx.ui.notify(`set_reward failed: ${(err as Error).message}`, "error");
      }
    },
  });

  pi.registerCommand("export_trajectories", {
    description:
      "Harvest a reward-stamped trajectory for one session (terminal: revokes it).",
    handler: async (args: string, ctx) => {
      const sessionId = args.trim();
      if (!sessionId) {
        ctx.ui.notify("usage: /export_trajectories <session_id>", "error");
        return;
      }
      try {
        await exportTrajectories(fetchImpl, gatewayConfigFromEnv(), { sessionId });
        ctx.ui.notify(`exported trajectory for session=${sessionId}`, "info");
      } catch (err) {
        ctx.ui.notify(`export failed: ${(err as Error).message}`, "error");
      }
    },
  });

  // Read-only transcript access -- the verifier judges from task_messages.
  pi.registerTool({
    name: "read_task_messages",
    label: "Read Task Messages",
    description:
      "Read (read-only) the Multica task_message transcript for a task_id, " +
      "ordered by seq. The verifier uses this to judge the collaboration.",
    promptSnippet: "Read the multi-agent task_message transcript for a task_id",
    parameters: {
      type: "object",
      properties: {
        task_id: { type: "string", description: "Multica task id to read" },
      },
      required: ["task_id"],
    },
    async execute(_toolCallId: string, params: { task_id: string }) {
      const baseUrl = process.env.MULTICA_BASE_URL ?? "";
      if (!baseUrl) throw new Error("MULTICA_BASE_URL is required");
      const headers: Record<string, string> = { Accept: "application/json" };
      const apiKey = process.env.MULTICA_API_KEY;
      if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
      const url = buildTaskMessagesUrl(baseUrl, params.task_id);
      const resp = await fetch(url, { method: "GET", headers });
      const text = await resp.text();
      if (!resp.ok) {
        throw new Error(
          `read_task_messages failed: status=${resp.status} body=${text.slice(0, 200)}`,
        );
      }
      return { content: [{ type: "text", text }], details: { url } };
    },
  });
}
