// src/config/jwtVerifier.js
//
// Local verification of Supabase access tokens (ES256) against the
// project's public keys (JWKS). No per-request call to Supabase.
//
//   verifySupabaseToken(token) -> claims   (sub, email, role, ...)
//     throws jose errors          -> token is bad          -> caller returns 401
//     throws AuthUnavailableError -> keys can't be loaded  -> caller returns 503
//
// Keys are fetched once (shortly after startup), refreshed in the background
// every hour, and re-fetched immediately when a token arrives with an unknown
// key id (Supabase key rotation), at most once every 30 s.

import { createLocalJWKSet, jwtVerify, errors } from "jose";

const BACKGROUND_REFRESH_MS = 60 * 60 * 1000; // 1 h
const MIN_REFETCH_GAP_MS = 30 * 1000;         // rate-limit for unknown-kid refetches
const FETCH_TIMEOUT_MS = 5000;
const CLOCK_TOLERANCE_S = 30;

export class AuthUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = "AuthUnavailableError";
  }
}

let keySet = null;       // jose key resolver built from the last good JWKS
let lastSuccess = 0;     // time of last successful fetch
let lastAttempt = 0;     // time of last fetch attempt (success or failure)
let inflight = null;     // shared promise so concurrent requests fetch once

// Read lazily: env may not be loaded yet when this module is imported.
function supabaseUrl() {
  const url = process.env.SUPABASE_URL;
  if (!url) throw new Error("SUPABASE_URL is not set");
  return url.replace(/\/+$/, "");
}

function fetchJwks() {
  if (inflight) return inflight;
  lastAttempt = Date.now();
  inflight = (async () => {
    const res = await fetch(`${supabaseUrl()}/auth/v1/.well-known/jwks.json`, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
    if (!res.ok) throw new Error(`JWKS request returned HTTP ${res.status}`);
    const jwks = await res.json();
    if (!Array.isArray(jwks.keys) || jwks.keys.length === 0) {
      throw new Error("JWKS response contains no keys");
    }
    keySet = createLocalJWKSet(jwks);
    lastSuccess = Date.now();
    console.log(`[auth] JWKS loaded (${jwks.keys.length} key(s))`);
  })().finally(() => {
    inflight = null;
  });
  return inflight;
}

async function ensureKeys() {
  if (!keySet) {
    try {
      await fetchJwks();
    } catch (err) {
      throw new AuthUnavailableError(`Could not load signing keys: ${err.message}`);
    }
    return;
  }
  // Have keys. If they're old, refresh in the background but keep using them.
  if (Date.now() - lastSuccess > BACKGROUND_REFRESH_MS && !inflight) {
    fetchJwks().catch((err) =>
      console.warn("[auth] background JWKS refresh failed (keeping cached keys):", err.message)
    );
  }
}

function verifyWithCurrentKeys(token) {
  const url = supabaseUrl();
  return jwtVerify(token, keySet, {
    issuer: `${url}/auth/v1`,
    audience: "authenticated",
    algorithms: ["ES256"],
    clockTolerance: CLOCK_TOLERANCE_S,
  });
}

export async function verifySupabaseToken(token) {
  await ensureKeys();
  try {
    const { payload } = await verifyWithCurrentKeys(token);
    return payload;
  } catch (err) {
    // Unknown key id: Supabase may have rotated keys. Re-fetch once and retry.
    const unknownKid = err instanceof errors.JWKSNoMatchingKey;
    if (unknownKid && Date.now() - lastAttempt > MIN_REFETCH_GAP_MS) {
      try {
        await fetchJwks();
      } catch (fetchErr) {
        throw new AuthUnavailableError(`Could not refresh signing keys: ${fetchErr.message}`);
      }
      const { payload } = await verifyWithCurrentKeys(token);
      return payload;
    }
    throw err; // jose error -> genuinely bad token
  }
}

// Load keys right after startup so the first request doesn't pay the
// network round trip. setImmediate lets the app finish loading env first.
setImmediate(() => {
  fetchJwks().catch((err) =>
    console.warn("[auth] JWKS warm-up failed, will retry on first request:", err.message)
  );
});