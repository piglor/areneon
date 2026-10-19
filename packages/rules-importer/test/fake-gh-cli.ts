// The end-to-end image's `gh` (through the test/e2e-bin/gh wrapper): answers
// from the recordings in fake-gh.ts. Test-only; never in the production image.

import { fakeGh } from "./fake-gh.ts";

const { code, stdout, stderr } = fakeGh(process.argv.slice(2));
process.stdout.write(stdout);
process.stderr.write(stderr);
process.exitCode = code ?? 1;
