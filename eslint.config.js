import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["**/dist/**", "**/coverage/**", "vendor/**", ".agents/**", "node_modules/**"] },
  js.configs.recommended,
  ...tseslint.configs.strict,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { ignoreRestSiblings: true }],
    },
  },
  // Spikes run plain JS in browsers, extension worlds and Node (lighter gates).
  {
    files: ["spikes/**/*.{js,cjs,mjs}"],
    languageOptions: {
      globals: Object.fromEntries(
        [
          "window", "document", "location", "crypto", "TextEncoder", "URL", "URLSearchParams",
          "WebSocket", "fetch", "setTimeout", "console", "process", "require", "__dirname", "global",
          "MessagePort", "MessageChannel", "EventTarget",
        ].map((name) => [name, "readonly"]),
      ),
    },
  },
  {
    files: ["spikes/**/*.cjs"],
    languageOptions: { sourceType: "commonjs" },
    rules: { "@typescript-eslint/no-require-imports": "off" },
  },
);
