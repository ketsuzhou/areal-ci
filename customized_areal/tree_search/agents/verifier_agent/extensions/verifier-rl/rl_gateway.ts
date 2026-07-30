/**
 * Gateway helpers for the pi verifier agent.
 *
 * The verifier agent reviews a finished multi-agent task and assigns a reward
 * per RL ``session_id`` (design decision 3). It talks to the AReaL inference
 * gateway via two admin-key endpoints:
 *
 *   - ``POST /rl/set_reward``        body ``{ session_id, reward }``
 *   - ``POST /export_trajectories``  body ``{ session_id }``  (terminal: the
 *                                    gateway revokes the session on success)
 *
 * and reads (read-only) the Multica ``task_message`` transcript it judges from.
 *
 * Everything here is framework-free (no pi import) so it is unit-testable with
 * a mock ``fetch``. The pi command registration lives in ``index.ts``.
 *
 * SECURITY: the verifier runs on a fixed judge model and MUST NOT route its own
 * LLM calls through ``areal/...``; these helpers only call the RL control-plane
 * endpoints with the admin key, never the chat-completions path.
 */

export interface GatewayConfig {
  /** Base URL of the AReaL inference gateway, e.g. http://gateway:8080 */
  baseUrl: string;
  /** Admin API key (Bearer) -- required for /rl/set_reward and /export_trajectories. */
  adminApiKey: string;
}

export interface SetRewardArgs {
  sessionId: string;
  reward: number;
}

export interface ExportTrajectoriesArgs {
  sessionId: string;
}

/** Minimal fetch shape so tests can inject a mock without DOM/node typings. */
export type FetchLike = (
  url: string,
  init: {
    method: string;
    headers: Record<string, string>;
    body: string;
  },
) => Promise<{
  status: number;
  text: () => Promise<string>;
}>;

export class GatewayError extends Error {
  readonly status: number;
  readonly path: string;
  readonly body: string;

  constructor(status: number, path: string, body: string) {
    super(`gateway ${path} failed: status=${status} body=${body.slice(0, 200)}`);
    this.name = "GatewayError";
    this.status = status;
    this.path = path;
    this.body = body;
  }
}

function trimTrailingSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Build the request for ``POST /rl/set_reward``. Pure -- no I/O. */
export function buildSetRewardRequest(
  config: GatewayConfig,
  args: SetRewardArgs,
): { url: string; method: string; headers: Record<string, string>; body: string } {
  if (!args.sessionId) {
    throw new Error("set_reward requires a non-empty session_id");
  }
  if (!Number.isFinite(args.reward)) {
    throw new Error(`set_reward requires a finite reward, got ${args.reward}`);
  }
  return {
    url: `${trimTrailingSlash(config.baseUrl)}/rl/set_reward`,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.adminApiKey}`,
    },
    body: JSON.stringify({ session_id: args.sessionId, reward: args.reward }),
  };
}

/** Build the request for ``POST /export_trajectories``. Pure -- no I/O. */
export function buildExportTrajectoriesRequest(
  config: GatewayConfig,
  args: ExportTrajectoriesArgs,
): { url: string; method: string; headers: Record<string, string>; body: string } {
  if (!args.sessionId) {
    throw new Error("export_trajectories requires a non-empty session_id");
  }
  return {
    url: `${trimTrailingSlash(config.baseUrl)}/export_trajectories`,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.adminApiKey}`,
    },
    body: JSON.stringify({ session_id: args.sessionId }),
  };
}

async function postJson(
  fetchImpl: FetchLike,
  req: { url: string; method: string; headers: Record<string, string>; body: string },
  path: string,
): Promise<unknown> {
  const resp = await fetchImpl(req.url, {
    method: req.method,
    headers: req.headers,
    body: req.body,
  });
  const text = await resp.text();
  if (resp.status < 200 || resp.status >= 300) {
    throw new GatewayError(resp.status, path, text);
  }
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

/** Set the reward for one RL session. Throws ``GatewayError`` on non-2xx. */
export async function setReward(
  fetchImpl: FetchLike,
  config: GatewayConfig,
  args: SetRewardArgs,
): Promise<unknown> {
  return postJson(fetchImpl, buildSetRewardRequest(config, args), "/rl/set_reward");
}

/**
 * Export (and revoke) one RL session's reward-stamped trajectory. Terminal --
 * the gateway revokes the session on a 200. Throws ``GatewayError`` on non-2xx.
 */
export async function exportTrajectories(
  fetchImpl: FetchLike,
  config: GatewayConfig,
  args: ExportTrajectoriesArgs,
): Promise<unknown> {
  return postJson(
    fetchImpl,
    buildExportTrajectoriesRequest(config, args),
    "/export_trajectories",
  );
}

/** Read gateway config from the environment (no secrets hardcoded). */
export function gatewayConfigFromEnv(
  env: Record<string, string | undefined> = process.env,
): GatewayConfig {
  const baseUrl = env.AREAL_GATEWAY_URL ?? "";
  const adminApiKey = env.AREAL_ADMIN_API_KEY ?? "";
  if (!baseUrl) throw new Error("AREAL_GATEWAY_URL is required");
  if (!adminApiKey) throw new Error("AREAL_ADMIN_API_KEY is required");
  return { baseUrl, adminApiKey };
}
