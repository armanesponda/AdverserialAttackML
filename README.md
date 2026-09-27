# Books Under the Sun

A social book-catalogue app: search a live catalogue (Open Library), keep a
personal shelf, rate and review what you've read, and see what your friends
are reading. Built with Next.js's App Router, a Postgres database via
Prisma, and Google sign-in via Auth.js.

## Stack

### Framework — Next.js 16 (App Router)

The whole app is one Next.js project — pages, API routes, and server-side
mutations all live under `src/app`. A few things worth knowing about this
specific setup:

- **Server Components by default.** Most page files (`src/app/**/page.tsx`)
  are `async` React Server Components that query Postgres directly and
  render HTML on the server — there's no separate REST/GraphQL API layer
  between the UI and the database for reads.
- **Server Actions for mutations.** `src/lib/actions.ts` exports plain
  `async` functions marked `"use server"`; client components import and
  call them directly (e.g. a button's `onClick`) instead of hand-rolling
  `fetch` calls to an API route. Next.js turns each one into its own POST
  endpoint under the hood. Every action re-checks who's signed in itself,
  since a Server Action is a public endpoint once shipped, not just a
  function only your own UI can invoke.
- **Streaming with `<Suspense>`.** The home page (`src/app/page.tsx`) splits
  into several independent async components — one per "shelf" — each behind
  its own `<Suspense>` boundary with a matching skeleton. The page shell
  renders immediately; each shelf streams in independently as its own data
  (often a third-party API call) resolves, rather than the whole page
  blocking on the slowest one.
- **"Proxy" instead of "Middleware."** `src/proxy.ts` is this project's
  version of what most Next.js docs still call Middleware — it runs before
  nearly every request and redirects any signed-in user without a username
  yet to `/welcome`. See `AGENTS.md` for why the naming/behavior here
  differs from what you might expect from general Next.js knowledge — this
  fork changed the convention, and it runs on the Node.js runtime rather
  than the Edge runtime, so it can do a real database session lookup rather
  than just an optimistic cookie check.
- **Route Handlers.** `src/app/api/auth/[...nextauth]/route.ts` is the one
  plain API route in the app — Auth.js's own catch-all endpoint.

### Language — TypeScript

Strict-mode TypeScript throughout, including a `src/types/next-auth.d.ts`
declaration-merging file that extends Auth.js's default session type with
this app's own `id`/`username` fields.

### Database — Postgres, via Prisma 7

- **Schema**: `prisma/schema.prisma` defines every table — Auth.js's
  required `User`/`Account`/`Session`/`VerificationToken` shape, plus this
  app's own `Book` (a local cache of Open Library data), `Subject`/
  `BookSubject`, `WatchlistItem`, `Rating`, `Post`, `Like`, and `Friendship`.
- **Driver adapter architecture**: Prisma 7 talks to Postgres through an
  explicit adapter (`@prisma/adapter-pg`, wrapping `node-postgres`) rather
  than its own bundled query engine — see `src/lib/db.ts`, where the
  `PrismaClient` is constructed with that adapter and reused across
  hot-reloads in dev via a `globalThis` cache (a standard Next.js pattern to
  avoid opening a fresh connection pool on every file save).
- **Hand-written migrations**: schema changes in `prisma/migrations/` are
  hand-written SQL files applied with `prisma migrate deploy`, rather than
  generated via `prisma migrate dev`/`diff` — this project has no shadow
  database configured, which those commands require.
- **Hosted on Neon** (Postgres-compatible serverless Postgres). Any standard
  Postgres host works — see `.env.example`.

### Auth — Auth.js (next-auth v5) + Google OAuth

`src/auth.ts` configures Auth.js with the Google provider, `@auth/
prisma-adapter` for persistence, and **database-backed sessions** (a real
`Session` row looked up per request, not a signed JWT) — chosen so a
session could be revoked server-side later without extra plumbing. The
`auth()` call is wrapped in React's `cache()` so multiple places that need
"who's signed in" within one request (a page and the header, say) share a
single session lookup instead of querying twice.

Accounts are Google-only for now; the app deliberately never surfaces a
signed-in user's real name or email to *other* users — see the `username`
system below.

### Styling — Tailwind CSS v4

Utility-first styling via `@tailwindcss/postcss`, no component library. The
design is a "library card catalogue" aesthetic — custom color tokens
(`paper`, `card`, `ink`, `stamp`, `due`, `rule`, …) and three Google Fonts
loaded via `next/font/google` (Space Grotesk for display type, IBM Plex Sans
for body text, IBM Plex Mono for the typewritten "label" text used for
categories/metadata) — defined in `src/app/globals.css` and `src/app/
layout.tsx`. Single light theme by design; the paper metaphor doesn't have
an obvious dark-mode equivalent.

### External data — Open Library API

`src/lib/open-library.ts` is a from-scratch client for the free,
keyless [Open Library API](https://openlibrary.org/developers/api) — no
book/author database of our own; every search, cover image, and category
listing is fetched live and normalized into this app's own `BookSummary`/
`BookDetail` shapes. Includes its own retry/timeout handling, a
duplicate-edition collapsing pipeline (Open Library indexes every
translation/printing of a work as a separate result), and relevance-ranking
logic layered on top of Open Library's own search ranking. A book is only
persisted into the local Postgres cache (the `Book` table) the first time a
user actually adds/rates/reviews it — browsing pages always read live.

### Edge relay — Cloudflare Worker

`cloudflare/ol-proxy/` is a small standalone Cloudflare Worker (plain
JavaScript, deployed with `wrangler`) that relays Open Library requests on
Vercel's behalf. It exists because Vercel's serverless functions (AWS IP
ranges) have their outbound connections to `openlibrary.org` silently
dropped in production — routing through Cloudflare's network sidesteps
that block. Only used in production (`OL_PROXY_URL`/`OL_PROXY_SECRET`); see
`cloudflare/ol-proxy/README.md` for deploy steps.

### Hosting — Vercel

Deployed on Vercel. `vercel.json` pins serverless functions to a single
region (see "Function region" below) to sit next to the database.

## Notable application-level design choices

- **Username-based privacy.** Accounts are identified internally by Google
  account (email), but friend search and profile URLs (`/u/[username]`) use
  a separate, self-chosen, nullable-until-set `username` — so finding or
  being found by another user never exposes a real name or email address.
  Every signed-in user without one yet is redirected to `/welcome` until
  they pick one (enforced in `src/proxy.ts`).
- **Request/accept friendships with race-condition safety.** Adding a
  friend creates a `Friendship` row that's `PENDING` until accepted.
  Concurrent/mutual friend requests (and double-clicks) are made safe with a
  Postgres advisory lock (`pg_advisory_xact_lock`) taken inside a
  transaction, keyed by the sorted user pair — the same pattern is reused
  for toggling a like on a review.
- **Per-shelf data scoping.** Reviews and ratings shown on a book's card are
  always scoped to whoever's shelf is being viewed (`where: { userId }` on
  every relevant query) — a shared cached `Book` row never leaks one user's
  review onto another user's shelf for the same book.
- **Friends-only shelf visibility**, with liking extended to reviews on a
  friend's shelf under the same authorization rule that gates seeing the
  shelf at all (`canViewShelf` in `src/lib/friends.ts`).
- **Recommendations** (`src/lib/recommendations.ts`) are computed directly
  from a user's own rating history — no external ML service — blending a
  subject/genre signal, an author signal, a recency-decay weighting (older
  ratings count for less), and a simple collaborative-filtering pass
  ("readers who rated your favorites highly also loved…") computed entirely
  from this app's own `Rating` table.

## Getting started

```bash
npm install
cp .env.example .env   # fill in DATABASE_URL, AUTH_SECRET, Google OAuth credentials
npx prisma generate
npx prisma migrate deploy
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). See `.env.example` for
what each environment variable is for and where to get it (Google OAuth
credentials, a Postgres connection string, an `AUTH_SECRET`).

## Scripts

- `npm run dev` — start the dev server
- `npm run build` / `npm run start` — production build and serve
- `npm run lint` — ESLint (`eslint-config-next`, flat config)
- `npx tsc --noEmit` — typecheck without emitting output
- `npx prisma migrate deploy` — apply pending migrations
- `npx prisma generate` — regenerate the Prisma Client into
  `src/generated/prisma` (also runs automatically via `postinstall`)

## Deployment

The easiest way to deploy this app is [Vercel](https://vercel.com), from the
creators of Next.js — see the [Next.js deployment
docs](https://nextjs.org/docs/app/building-your-application/deploying) for
the general flow. Two things specific to this project beyond a standard
Next.js deploy:

- Set the Cloudflare Worker relay env vars (`OL_PROXY_URL`/
  `OL_PROXY_SECRET`) — see `cloudflare/ol-proxy/README.md` — or Open Library
  calls will be silently blocked in production.
- Set `AUTH_GOOGLE_ID`/`AUTH_GOOGLE_SECRET`/`AUTH_SECRET`/`DATABASE_URL` per
  `.env.example`, and add the deployed URL's `/api/auth/callback/google` as
  an authorized redirect URI in the Google Cloud Console.

### Function region

`vercel.json` pins serverless functions to `cle1` (Cleveland, Ohio / AWS
`us-east-2`) to match the Neon Postgres database's region — nearly every
route here makes at least one DB round trip, so collocating the two removes
a cross-region hop on every request. This is a single-region pin, not an
edge deployment: every invocation, from any visitor anywhere, runs in Ohio.
That's a deliberate trade — DB-adjacent latency matters more than global
reach for this app's current (US-based) user base. If the user base becomes
meaningfully global, revisit this rather than reflexively "fixing" a
slow-for-distant-users complaint by removing it.
