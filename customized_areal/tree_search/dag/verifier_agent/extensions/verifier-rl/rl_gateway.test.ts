/**
 * Tests for the pi verifier gateway helpers.
 *
 * Run: node --test rl_gateway.test.ts   (Node >= 23 strips TS types natively)
 *
 * Covers: correct request bodies for /rl/set_reward and /export_trajectories,
 * a mock gateway asserting the forwarded payload, and HTTP error surfacing.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildExportTrajectoriesRequest,
  buildSetRewardRequest,
  exportTrajectories,
  GatewayError,
  gatewayConfigFromEnv,
  setReward,
  type FetchLike,
} from "./rl_gateway.ts";

const CONFIG = { baseUrl: "http://gw:8080/", adminApiKey: "admin-key" };

test("buildSetRewardRequest produces the {session_id, reward} body", () => {
  const req = buildSetRewardRequest(CONFIG, { sessionId: "S", reward: 0.8 });
  assert.equal(req.url, "http://gw:8080/rl/set_reward"); // trailing slash trimmed
  assert.equal(req.method, "POST");
  assert.equal(req.headers.Authorization, "Bearer admin-key");
  assert.deepEqual(JSON.parse(req.body), { session_id: "S", reward: 0.8 });
});

test("buildExportTrajectoriesRequest produces the {session_id} body", () => {
  const req = buildExportTrajectoriesRequest(CONFIG, { sessionId: "S" });
  assert.equal(req.url, "http://gw:8080/export_trajectories");
  assert.deepEqual(JSON.parse(req.body), { session_id: "S" });
});

test("set_reward rejects empty session and non-finite reward", () => {
  assert.throws(() => buildSetRewardRequest(CONFIG, { sessionId: "", reward: 1 }));
  assert.throws(() =>
    buildSetRewardRequest(CONFIG, { sessionId: "S", reward: Number.NaN }),
  );
});

test("setReward posts the expected payload to the mock gateway", async () => {
  const calls: Array<{ url: string; body: string }> = [];
  const fetchImpl: FetchLike = async (url, init) => {
    calls.push({ url, body: init.body });
    return { status: 200, text: async () => JSON.stringify({ ok: true }) };
  };
  const out = await setReward(fetchImpl, CONFIG, { sessionId: "sess-1", reward: 0.5 });
  assert.deepEqual(out, { ok: true });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://gw:8080/rl/set_reward");
  assert.deepEqual(JSON.parse(calls[0].body), { session_id: "sess-1", reward: 0.5 });
});

test("exportTrajectories posts {session_id} and parses JSON", async () => {
  const calls: string[] = [];
  const fetchImpl: FetchLike = async (url, init) => {
    calls.push(init.body);
    return { status: 200, text: async () => JSON.stringify({ interactions: {} }) };
  };
  const out = await exportTrajectories(fetchImpl, CONFIG, { sessionId: "sess-2" });
  assert.deepEqual(out, { interactions: {} });
  assert.deepEqual(JSON.parse(calls[0]), { session_id: "sess-2" });
});

test("non-2xx surfaces a GatewayError with status and body", async () => {
  const fetchImpl: FetchLike = async () => ({
    status: 404,
    text: async () => "session not found",
  });
  await assert.rejects(
    () => setReward(fetchImpl, CONFIG, { sessionId: "missing", reward: 1 }),
    (err: unknown) => {
      assert.ok(err instanceof GatewayError);
      assert.equal(err.status, 404);
      assert.equal(err.path, "/rl/set_reward");
      assert.match(err.message, /session not found/);
      return true;
    },
  );
});

test("gatewayConfigFromEnv requires both env vars", () => {
  assert.throws(() => gatewayConfigFromEnv({ AREAL_GATEWAY_URL: "http://x" }));
  assert.throws(() => gatewayConfigFromEnv({ AREAL_ADMIN_API_KEY: "k" }));
  const cfg = gatewayConfigFromEnv({
    AREAL_GATEWAY_URL: "http://x",
    AREAL_ADMIN_API_KEY: "k",
  });
  assert.deepEqual(cfg, { baseUrl: "http://x", adminApiKey: "k" });
});
