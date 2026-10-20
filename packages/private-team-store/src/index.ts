// The private team store (ADR-014). This package is deliberately empty of
// features: it exists so the packaging rule (test/packaging.test.ts) can hold
// from the first commit. It is never shipped: no image, Coolify build or other
// workspace package may depend on it.
export const PRIVATE_TEAM_STORE_PACKAGE = "@areneon/private-team-store";
