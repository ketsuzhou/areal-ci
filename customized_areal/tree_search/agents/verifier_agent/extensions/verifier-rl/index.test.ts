/**
 * Tests for the pi verifier command parsing + transcript URL builder.
 *
 * Run: node --test index.test.ts
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { buildTaskMessagesUrl, parseSetRewardArgs } from "./index.ts";

test("parseSetRewardArgs parses '<session> <reward>'", () => {
  assert.deepEqual(parseSetRewardArgs("sess-1 0.8"), {
    sessionId: "sess-1",
    reward: 0.8,
  });
  assert.deepEqual(parseSetRewardArgs("  sess-2   1  "), {
    sessionId: "sess-2",
    reward: 1,
  });
});

test("parseSetRewardArgs rejects wrong arity and non-numeric reward", () => {
  assert.throws(() => parseSetRewardArgs("only-session"));
  assert.throws(() => parseSetRewardArgs("a b c"));
  assert.throws(() => parseSetRewardArgs("sess notanumber"));
});

test("buildTaskMessagesUrl encodes the task id and trims slashes", () => {
  assert.equal(
    buildTaskMessagesUrl("http://multica/", "task 1"),
    "http://multica/api/tasks/task%201/messages",
  );
  assert.throws(() => buildTaskMessagesUrl("http://multica", ""));
});
