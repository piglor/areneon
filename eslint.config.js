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
);
