/**
 * style.css must parse cleanly. A stray declaration or brace does not fail the
 * page -- CSS error recovery silently swallows the NEXT rule, which is how a
 * leftover fragment once ate `.hero { ... }` and broke the homepage intro.
 */
import { expect, test } from "bun:test";
import { resolve } from "node:path";

test("style.css parses without errors or warnings", async () => {
  const result = await Bun.build({
    entrypoints: [resolve(import.meta.dir, "..", "style.css")],
    throw: false,
  });
  const problems = [...result.logs].map((l) => `${l.level}: ${l.message}`);
  expect(problems, problems.join("\n")).toHaveLength(0);
  expect(result.success).toBe(true);
});
