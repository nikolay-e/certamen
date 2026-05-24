# Certamen GUI (React + TypeScript + Vite)

Web frontend for the Certamen workflow editor. Built with React 19, Vite, and
`@xyflow/react` for the node graph canvas.

## Commands

```bash
npm run dev         # Vite dev server with HMR
npm run build       # tsc -b && vite build
npm run preview     # preview the production build
npm run type-check  # tsc --noEmit
npm run lint        # biome check (lint + format + import sort)
npm run lint:fix    # biome check --write (apply safe fixes + format)
npm run format      # biome format --write
npm run test:e2e    # Playwright end-to-end tests
```

## Tooling

Linting, formatting, and import sorting are handled by a single tool,
[Biome](https://biomejs.dev/) (`biome.json`), which replaces ESLint and
Prettier. The linter runs Biome's `recommended` rules plus the `react` domain
(rules of hooks, exhaustive dependencies). Justified per-line exceptions use
`// biome-ignore <rule>: <reason>` comments rather than disabling rules globally.
