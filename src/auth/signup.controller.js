// src/auth/signup.controller.js
//
// ONB-4 (2026-11-15) auth routes for the new first-time experience.
//
// New endpoints use the envelope  {success: true, data}  /
//                                 {success: false, error: {code, message, ...}}
//   GET  /api/auth/email-available?email=   -> {available}
//   POST /api/auth/signup/start    {name, email, password, birthday}  -> {pending_id, resend_after, expires_in}
//   POST /api/auth/signup/resend   {pending_id}                       -> {resend_after, expires_in}
//   POST /api/auth/signup/verify   {pending_id, code}                 -> {verified: true}
//   POST /api/auth/signup/complete {pending_id, terms_version, password}
//                                  -> 201 {message, user, access_token, refresh_token, is_new_user}
//
// Login and forgot password keep their old success bodies (older app versions):
//   POST /api/auth/login  F2: wrong details -> 401 {error, code: LOGIN_FAILED} (never says which part)
//                         5 wrong in 15 min -> 429 {error, code: LOGIN_LOCKED, retry_after} for 5 min
//   POST /api/auth/forgot-password  F3: always the same 200 answer, whether or not the email exists

import redis from "../config/redis.js";
import authService, { forgotPassword as forgotPasswordService } from "./auth.service.js";
import authValidation from "./auth.validation.js";
import * as signup from "./signup.service.js";

const LOCK_AFTER = 5;          // wrong passwords ...
const FAIL_WINDOW_S = 900;     // ... within 15 minutes
const LOCK_S = 300;            // -> locked for 5 minutes

function ok(res, data, status = 200) {
  return res.status(status).json({ success: true, data });
}

function fail(res, err) {
  if (err instanceof signup.SignupError) {
    if (err.extra?.retry_after) res.set("Retry-After", String(err.extra.retry_after));
    if (err.code === "DB_UNAVAILABLE" || err.code === "EMAIL_UNAVAILABLE") res.set("Retry-After", "5");
    return res.status(err.status).json({
      success: false, error: { code: err.code, message: err.message, ...err.extra },
    });
  }
  console.error("[signup] unexpected error:", err);
  return res.status(500).json({
    success: false, error: { code: "INTERNAL_ERROR", message: "Something went wrong. Please try again." },
  });
}

const wrap = (fn) => async (req, res) => {
  try {
    await fn(req, res);
  } catch (err) {
    fail(res, err);
  }
};

// ---------------------------------------------------------------- A

export const emailAvailable = wrap(async (req, res) => ok(res, await signup.emailAvailable(req.query.email)));
export const start = wrap(async (req, res) => ok(res, await signup.start(req.body || {})));
export const resend = wrap(async (req, res) => ok(res, await signup.resend(req.body || {})));
export const verify = wrap(async (req, res) => ok(res, await signup.verify(req.body || {})));
export const complete = wrap(async (req, res) => ok(res, await signup.complete(req.body || {}), 201));

// ---------------------------------------------------------------- F2 login with lockout

const failKey = (email) => `login_fail:${email}`;
const lockKey = (email) => `login_lock:${email}`;

async function lockedFor(email) {
  try {
    const ttl = await redis.ttl(lockKey(email));
    return ttl > 0 ? ttl : 0;
  } catch {
    return 0;                       // redis down: never block a login because of it
  }
}

async function countFailure(email) {
  try {
    const n = await redis.incr(failKey(email));
    if (n === 1) await redis.expire(failKey(email), FAIL_WINDOW_S);
    if (n >= LOCK_AFTER) {
      await redis.set(lockKey(email), "1", { ex: LOCK_S });
      await redis.del(failKey(email));
      return true;
    }
  } catch (err) {
    console.warn("[login] failure count not saved:", err.message);
  }
  return false;
}

function lockedReply(res, seconds) {
  res.set("Retry-After", String(seconds));
  const minutes = Math.max(1, Math.ceil(seconds / 60));
  return res.status(429).json({
    error: `Too many tries. Wait ${minutes} minute${minutes === 1 ? "" : "s"} and try again.`,
    code: "LOGIN_LOCKED",
    retry_after: seconds,
  });
}

export const login = async (req, res) => {
  const errors = authValidation.validateLogin(req.body || {});
  if (Object.keys(errors).length > 0) {
    return res.status(400).json({ error: "Validation failed", errors });
  }
  const email = String(req.body.email).trim().toLowerCase();

  const wait = await lockedFor(email);
  if (wait) return lockedReply(res, wait);

  try {
    const result = await authService.loginUser({ email, password: req.body.password });
    try { await redis.del(failKey(email)); } catch { /* best effort */ }
    return res.json({
      message: "Login successful",
      user: { id: result.user.id, email: result.user.email },
      access_token: result.access_token,
      refresh_token: result.refresh_token,
    });
  } catch (error) {
    const msg = error.message || "";
    if (msg.startsWith("Incorrect email or password")) {
      if (await countFailure(email)) return lockedReply(res, LOCK_S);
      return res.status(401).json({ error: "Incorrect email or password.", code: "LOGIN_FAILED" });
    }
    if (msg.startsWith("Please confirm your email")) {
      return res.status(403).json({ error: msg, code: "EMAIL_NOT_CONFIRMED" });
    }
    if (msg.startsWith("Too many login attempts")) {
      return res.status(429).json({ error: msg, code: "RATE_LIMITED" });
    }
    return res.status(400).json({ error: msg || "Something went wrong logging in. Please try again." });
  }
};

// ---------------------------------------------------------------- F3 forgot password

export const forgotPassword = async (req, res) => {
  const email = String(req.body?.email ?? "").trim().toLowerCase();
  if (!email || !email.includes("@")) {
    return res.status(400).json({ error: "Invalid email address" });
  }
  try {
    await forgotPasswordService(email);
  } catch (error) {
    // Never tell the caller: the answer must be the same whether or not the email exists.
    console.warn("[forgot-password] not sent for", email, ":", error.message);
  }
  return res.json({ message: "If there's an account with this email, we've sent a code to it." });
};

const signupController = { emailAvailable, start, resend, verify, complete, login, forgotPassword };
export default signupController;