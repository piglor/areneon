# The rules importer (ADR-007 §6, #768). Build from the repository root:
#
#   docker build -f packages/rules-importer/Dockerfile -t areneon-rules-importer .
#
# Runtime configuration comes only from the environment: DATABASE_URL (the
# `importer` role; `sslmode=verify-full` in production) and GH_TOKEN (read-only,
# for gh), plus the optional RULES_IMPORT_*_TIMEOUT_MS session timeouts
# (see src/cli.ts). Nothing secret is baked in.
#
# Targets: `runtime` (the default, last stage) is the production image. `e2e`
# is that image plus a stand-in gh that replays recorded GitHub answers and the
# recorded release fixtures, for the CI end-to-end import; it is never shipped,
# and the production image contains neither the stand-in nor the fixtures.

ARG NODE_IMAGE=node:24.22.0-bookworm-slim@sha256:a5f4366e834695d5f363b920c104953a535d3a9864d1bdf74566a0ebe01daee5
# The runtime base: the Debian release the Node image is built on.
ARG DEBIAN_IMAGE=debian:bookworm-20261005-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

# gh from its official release tarball, checked against a pinned SHA-256.
FROM ${NODE_IMAGE} AS gh
ARG TARGETARCH
ARG GH_VERSION=2.102.0
ARG GH_SHA256_AMD64=bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386
ARG GH_SHA256_ARM64=7862c86c72f43df3a2d93ddde6f473285b4e2af61b494849846827e513ef6484
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
  case "${TARGETARCH}" in \
    amd64) sum="${GH_SHA256_AMD64}" ;; \
    arm64) sum="${GH_SHA256_ARM64}" ;; \
    *) echo "unsupported architecture ${TARGETARCH}" >&2; exit 1 ;; \
  esac; \
  name="gh_${GH_VERSION}_linux_${TARGETARCH}"; \
  curl -fsSL -o "/tmp/${name}.tar.gz" "https://github.com/cli/cli/releases/download/v${GH_VERSION}/${name}.tar.gz"; \
  echo "${sum}  /tmp/${name}.tar.gz" | sha256sum -c -; \
  tar -xzf "/tmp/${name}.tar.gz" -C /tmp "${name}/bin/gh"; \
  install -m 0755 "/tmp/${name}/bin/gh" /usr/local/bin/gh; \
  gh --version

# The application: production dependencies of the importer and the workspace
# packages it uses, then their sources. Workspace packages stay symlinks to
# packages/*, outside node_modules, so Node strips their TypeScript as it does
# in the repository. pnpm and corepack stay in this stage.
FROM ${NODE_IMAGE} AS build
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/rules-importer/package.json packages/rules-importer/
COPY packages/rules-store/package.json packages/rules-store/
COPY packages/showdown-bridge/package.json packages/showdown-bridge/
COPY packages/schema/package.json packages/schema/
RUN corepack enable \
  && pnpm install --frozen-lockfile --prod --ignore-scripts --filter "@areneon/rules-importer..."
COPY packages/rules-importer/src packages/rules-importer/src
COPY packages/rules-store/src packages/rules-store/src
COPY packages/showdown-bridge/src packages/showdown-bridge/src
COPY packages/schema/src packages/schema/src
COPY packages/schema/rules packages/schema/rules
COPY packages/schema/contracts packages/schema/contracts

# The runtime holds only node, gh, the CA bundle and /app: no apt step, no
# curl, npm, corepack or pnpm. The node binary needs only glibc and libstdc++,
# which Debian's slim image has.
FROM ${DEBIAN_IMAGE} AS base
COPY --from=gh /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=gh /usr/local/bin/gh /usr/local/bin/gh
COPY --from=build /usr/local/bin/node /usr/local/bin/node
COPY --from=build /app /app
# An unprivileged user whose home holds gh's cache.
RUN useradd --uid 1000 --user-group --create-home --shell /usr/sbin/nologin importer
WORKDIR /app
ENV NODE_ENV=production \
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
USER 1000:1000
ENTRYPOINT ["node", "packages/rules-importer/src/cli.ts"]

# Test-only: a gh earlier on PATH that replays the recorded GitHub answers.
FROM base AS e2e
COPY fixtures/rules-releases fixtures/rules-releases
COPY packages/rules-importer/test/fake-gh.ts packages/rules-importer/test/fake-gh-cli.ts packages/rules-importer/test/
COPY packages/rules-importer/test/e2e-bin packages/rules-importer/test/e2e-bin
ENV PATH=/app/packages/rules-importer/test/e2e-bin:${PATH}

# The production image (default target).
FROM base AS runtime
