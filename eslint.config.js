import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["**/dist/**", "**/coverage/**", "vendor/**", ".agents/**", "node_modules/**", ".local-server/**", "**/.venv/**"] },
  js.configs.recommended,
  ...tseslint.configs.strict,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { ignoreRestSiblings: true }],
    },
  },
  {
    // Sealed-set code handles the key, the salt and the sealed teams: it must not print anything. Output goes through the injected `io` of the command.
    files: ["packages/team-dataset/src/sealed*.ts"],
    rules: { "no-console": "error" },
  },
  {
    // The replay viewer is two classic scripts that Python inlines into one page: they share globals
    // (`ArReplay`, `ArView`) instead of importing, and read the page's own document and window.
    files: ["packages/replay-view/web/**/*.js"],
    languageOptions: { sourceType: "script", globals: { document: "readonly", window: "readonly" } },
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { ignoreRestSiblings: true, varsIgnorePattern: "^Ar(Replay|View)$" }],
      "@typescript-eslint/no-dynamic-delete": "off",
    },
  },
  {
    files: ["packages/replay-view/web/view.js"],
    languageOptions: { globals: { ArReplay: "readonly" } },
  },
  {
    // The replay tests drive the loaded scripts and a DOM stand-in: their values are untyped by nature.
    files: ["packages/replay-view/test/**/*.ts", "packages/replay-view/web/**/*.d.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "no-var": "off",
    },
  },
);
