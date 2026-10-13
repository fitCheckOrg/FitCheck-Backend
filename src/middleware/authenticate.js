// src/middleware/authenticate.js
//
// Token check is LOCAL (src/config/jwtVerifier.js), so a slow or failing
// network can no longer turn a valid token into a 401.
//
//   no token / bad signature / expired / wrong issuer  -> 401 AUTH_INVALID / AUTH_MISSING
//   signing keys can't be loaded                       -> 503 AUTH_UNAVAILABLE
//   unexpected server bug                              -> 500 AUTH_ERROR
//
// The profiles row is still merged into req.user (same shape as before:
// auth user fields + profile columns). It's cached in redis per USER for
// 15 min. If it can't be loaded, the request is NOT rejected: req.user has
// the token's identity, req.profileUnavailable = true, and only routes that
// need the profile (requirePremium, /auth/me) answer 503.

import redis from "../config/redis.js";
import supabase from "../config/supabase.js";
import { errors as joseErrors } from "jose";
import { verifySupabaseToken, AuthUnavailableError } from "../config/jwtVerifier.js";

const PROFILE_TTL_S = 900;
const REDIS_TIMEOUT_MS = 1500;
const PROFILE_TIMEOUT_MS = 5000;

const profileKey = (userId) => `profile:${userId}`;

function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

export function sendUnavailable(res, code = "AUTH_UNAVAILABLE",
                                message = "Authentication temporarily unavailable, please retry") {
  res.set("Retry-After", "5");
  return res.status(503).json({ error: message, code });
}

// Call this after any code that UPDATES a profiles row (tier change,
// name change, ...) so the next request doesn't see the 15-min-old copy.
export async function invalidateProfile(userId) {
  try {
    await withTimeout(redis.del(profileKey(userId)), REDIS_TIMEOUT_MS, "redis del");
  } catch (err) {
    console.warn("[auth] profile cache invalidation failed:", err.message);
  }
}

async function loadProfile(userId) {
  try {
    const cached = await withTimeout(redis.get(profileKey(userId)), REDIS_TIMEOUT_MS, "redis get");
    if (cached) {
      return { profile: typeof cached === "string" ? JSON.parse(cached) : cached, ok: true };
    }
  } catch {
    // redis slow/down: fall through to the database
  }

  try {
    const { data, error } = await withTimeout(
      supabase.from("profiles").select("*").eq("id", userId).maybeSingle(),
      PROFILE_TIMEOUT_MS,
      "profile lookup"
    );
    if (error) throw error;
    if (data) {
      Promise.resolve(redis.set(profileKey(userId), JSON.stringify(data), { ex: PROFILE_TTL_S }))
        .catch(() => {});
    }
    return { profile: data ?? null, ok: true }; // ok + null = no profile row exists
  } catch (err) {
    console.warn("[auth] profile lookup failed:", err.message);
    return { profile: null, ok: false };
  }
}

export const authenticate = async (req, res, next) => {
  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ") ? header.slice(7).trim() : "";

  if (!token) {
    return res.status(401).json({ error: "Unauthorized", code: "AUTH_MISSING" });
  }

  let claims;
  try {
    claims = await verifySupabaseToken(token);
  } catch (err) {
    if (err instanceof AuthUnavailableError) {
      console.warn("[auth] verification unavailable:", err.message);
      return sendUnavailable(res);
    }
    if (err instanceof joseErrors.JOSEError) {
      return res.status(401).json({ error: "Invalid or expired token", code: "AUTH_INVALID" });
    }
    console.error("AUTH MIDDLEWARE ERROR:", err);
    return res.status(500).json({ error: "Authentication error", code: "AUTH_ERROR" });
  }

  const { profile, ok } = await loadProfile(claims.sub);

  req.user = {
    id: claims.sub,
    email: claims.email,
    phone: claims.phone,
    role: claims.role,
    user_metadata: claims.user_metadata ?? {},
    app_metadata: claims.app_metadata ?? {},
    ...(profile ?? {}),
  };
  req.profile = profile;
  req.profileUnavailable = !ok;

  next();
};

export const requirePremium = (req, res, next) => {
  if (req.profileUnavailable) {
    return sendUnavailable(res, "PROFILE_UNAVAILABLE", "Account details temporarily unavailable, please retry");
  }
  if (req.user?.tier !== "premium") {
    return res.status(403).json({
      error: "Premium required",
      message: "Upgrade to FitCheck Premium to unlock this feature",
    });
  }
  next();
};

export const requireOwnership = (userIdParam = "userId") => {
  return (req, res, next) => {
    const resourceUserId = req.params[userIdParam];

    if (req.user.id !== resourceUserId) {
      return res.status(403).json({ error: "Access denied" });
    }
    next();
  };
};