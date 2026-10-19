// The supervisor process `launch` starts: see `supervise` in src/supervisor.ts.
// argv[2] is the JSON SupervisorConfig; stdin is held open by the launcher.
import { supervise, type SupervisorConfig } from "../src/supervisor.ts";

const config = JSON.parse(process.argv[2] ?? "") as SupervisorConfig;
const code = await supervise(config, {
  input: process.stdin,
  events: process.stdout,
  log: process.stderr,
  onSignal(handler) {
    for (const name of ["SIGINT", "SIGTERM", "SIGHUP"] as const) process.on(name, () => handler(name));
  },
});
process.exit(code);
