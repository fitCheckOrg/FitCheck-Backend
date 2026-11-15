// src/auth/signup.service.js
//
// ONB-4 A (2026-11-15): email sign-up with a 6-digit code + birthday.
// The account only exists after the code is verified AND terms are accepted.
//
//   start    -> checks name / email / password / birthday, emails a code, saves a
//               PENDING row (signup_pending). Under the minimum age -> nothing is saved.
//   resend   -> new code (30 s gap, at most MAX_RESENDS)
//   verify   -> checks the code (5 wrong tries -> locked, start again)
//   complete -> terms accepted -> creates the Supabase user + profile + private row,
//               signs in, returns the same body as login.
//
// Stored in the pending row: email, name, birthday, a salted scrypt hash of the
// password, an HMAC of the code. Never the password or the code themselves.
// Because only a hash is kept, `complete` needs the password again (the app
// still has it in memory; it never saves it).
//
// .env (optional):
//   MIN_SIGNUP_AGE=18          minimum age (Keji's decision: 18)
//   SIGNUP_CODE_SECRET=...     key for hashing codes (falls back to SUPABASE_SERVICE_KEY)

import crypto from "node:crypto";
import { createClient } from "@supabase/supabase-js";
import supabase from "../config/supabase.js";
import { sendWelcomeEmail } from "../config/emailService.js";

export const CODE_TTL_S = 600;          // a code lives 10 minutes
export const RESEND_AFTER_S = 30;       // wait between codes
const VERIFIED_TTL_S = 1800;            // after verify: 30 minutes to accept terms
const MAX_ATTEMPTS = 5;
const MAX_RESENDS = 5;
const BREVO_URL = "https://api.brevo.com/v3/smtp/email";
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export class SignupError extends Error {
  constructor(status, code, message, extra = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.extra = extra;
  }
}

const minAge = () => {
  const n = parseInt(process.env.MIN_SIGNUP_AGE || "18", 10);
  return Number.isFinite(n) && n > 0 ? n : 18;
};
const now = () => new Date();
const later = (seconds) => new Date(Date.now() + seconds * 1000).toISOString();
const secondsUntil = (iso) => Math.max(0, Math.ceil((new Date(iso).getTime() - Date.now()) / 1000));

// ---------------------------------------------------------------- checks

export function normEmail(raw) {
  const email = String(raw ?? "").trim().toLowerCase();
  if (!email || email.length > 254 || !EMAIL_RE.test(email)) {
    throw new SignupError(400, "EMAIL_INVALID", "Please enter a valid email address.");
  }
  return email;
}

export function checkName(raw) {
  const name = String(raw ?? "").split(/\s+/).filter(Boolean).join(" ");
  if (name.length < 1 || name.length > 50) {
    throw new SignupError(400, "NAME_INVALID", "Name must be 1-50 characters.");
  }
  return name;
}

export function checkPassword(raw, name, email) {
  const pw = typeof raw === "string" ? raw : "";
  const weak = (msg) => new SignupError(400, "PASSWORD_WEAK", msg);
  if (pw.length < 8) throw weak("Use at least 8 characters.");
  if (pw.length > 72) throw weak("Use 72 characters or fewer.");
  if (!/[0-9]|[^A-Za-z0-9]/.test(pw)) throw weak("Add a number or a symbol.");
  const low = pw.toLowerCase();
  const local = email.split("@")[0];
  const squashedName = name.toLowerCase().replace(/\s+/g, "");
  if (low.includes(email) || (local.length >= 3 && low.includes(local))) {
    throw weak("Your password can't contain your email.");
  }
  if (squashedName.length >= 3 && low.replace(/\s+/g, "").includes(squashedName)) {
    throw weak("Your password can't contain your name.");
  }
  return pw;
}

// "2001-04-09" -> age in whole years, or BIRTHDAY_INVALID
export function ageOn(raw, today = now()) {
  const s = String(raw ?? "");
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  const bad = () => new SignupError(400, "BIRTHDAY_INVALID", "Please enter a real date of birth.");
  if (!m) throw bad();
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const date = new Date(Date.UTC(y, mo - 1, d));
  if (date.getUTCFullYear() !== y || date.getUTCMonth() !== mo - 1 || date.getUTCDate() !== d) throw bad();
  if (date > today) throw bad();
  let age = today.getUTCFullYear() - y;
  const beforeBirthday = today.getUTCMonth() < mo - 1 ||
    (today.getUTCMonth() === mo - 1 && today.getUTCDate() < d);
  if (beforeBirthday) age -= 1;
  if (age > 120) throw bad();
  return age;
}

export function checkAge(birthday) {
  const age = ageOn(birthday);
  if (age < minAge()) {
    throw new SignupError(422, "UNDER_AGE", `You need to be ${minAge()} or older to use FitCheck.`,
      { min_age: minAge() });
  }
  return age;
}

// ---------------------------------------------------------------- hashing

const codeKey = () => process.env.SIGNUP_CODE_SECRET || process.env.SUPABASE_SERVICE_KEY || "fitcheck";

function hashCode(pendingId, code) {
  return crypto.createHmac("sha256", codeKey()).update(`${pendingId}:${code}`).digest("hex");
}

function hashPassword(pw) {
  const salt = crypto.randomBytes(16);
  const hash = crypto.scryptSync(pw, salt, 32);
  return `${salt.toString("hex")}:${hash.toString("hex")}`;
}

function passwordMatches(pw, stored) {
  const [saltHex, hashHex] = String(stored || "").split(":");
  if (!saltHex || !hashHex) return false;
  const hash = crypto.scryptSync(String(pw ?? ""), Buffer.from(saltHex, "hex"), 32);
  return crypto.timingSafeEqual(hash, Buffer.from(hashHex, "hex"));
}

function sameHex(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  return crypto.timingSafeEqual(Buffer.from(a, "hex"), Buffer.from(b, "hex"));
}

const newCode = () => String(crypto.randomInt(0, 1000000)).padStart(6, "0");

// ---------------------------------------------------------------- email

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function codeEmail(name, code) {
  return {
    subject: `${code} is your FitCheck code`,
    html: `
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0A0A0A; color: #F5F0EB; padding: 40px; border-radius: 12px;">
      <h1 style="font-size: 26px; font-weight: 700; margin-bottom: 8px;">Confirm your email</h1>
      <p style="font-size: 16px; line-height: 1.6; color: #D1D5DB;">Hi ${esc(name)}, here's your code to finish creating your FitCheck account:</p>
      <p style="font-size: 36px; font-weight: 700; letter-spacing: 8px; margin: 24px 0;">${code}</p>
      <p style="font-size: 14px; color: #9CA3AF;">It expires in 10 minutes. If you didn't try to sign up, you can ignore this email.</p>
      <hr style="border: none; border-top: 1px solid #1F2937; margin: 40px 0;" />
      <p style="font-size: 13px; color: #6B7280;">&copy; FitCheck</p>
    </div>`,
  };
}

// Unlike the other emails this one is AWAITED: without the code the user is stuck.
async function sendCode(email, name, code) {
  const apiKey = process.env.BREVO_API_KEY;
  const fromEmail = process.env.EMAIL_FROM;
  if (!apiKey || !fromEmail) {
    console.error("[signup] code email not sent: BREVO_API_KEY or EMAIL_FROM missing in .env");
    return false;
  }
  const t = codeEmail(name, code);
  try {
    const res = await fetch(BREVO_URL, {
      method: "POST",
      headers: { "api-key": apiKey, "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify({
        sender: { email: fromEmail, name: process.env.EMAIL_FROM_NAME || "FitCheck" },
        to: [{ email }],
        subject: t.subject,
        htmlContent: t.html,
      }),
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) {
      console.error(`[signup] code email to ${email} failed: HTTP ${res.status} ${await res.text().catch(() => "")}`);
      return false;
    }
    console.log(`[signup] code email sent to ${email}`);
    return true;
  } catch (err) {
    console.error(`[signup] code email to ${email} failed: ${err.message}`);
    return false;
  }
}

// ---------------------------------------------------------------- data

export async function emailTaken(email) {
  const pattern = email.replace(/[\\%_]/g, (c) => `\\${c}`);      // ilike: exact, any case
  const { data, error } = await supabase.from("profiles").select("id").ilike("email", pattern).limit(1);
  if (error) throw new SignupError(503, "DB_UNAVAILABLE", "We couldn't reach the server just now. Please try again.");
  return (data || []).length > 0;
}

async function getPending(pendingId) {
  if (!UUID_RE.test(String(pendingId ?? ""))) return null;
  const { data, error } = await supabase.from("signup_pending").select("*").eq("id", pendingId).maybeSingle();
  if (error) throw new SignupError(503, "DB_UNAVAILABLE", "We couldn't reach the server just now. Please try again.");
  if (!data) return null;
  if (new Date(data.expires_at) <= now()) {
    await supabase.from("signup_pending").delete().eq("id", data.id);
    return null;
  }
  return data;
}

const notFound = () =>
  new SignupError(404, "PENDING_NOT_FOUND", "This sign-up has expired. Please start again.");

// ---------------------------------------------------------------- A1 / A2-A5

export async function emailAvailable(rawEmail) {
  const email = normEmail(rawEmail);
  return { available: !(await emailTaken(email)) };
}

export async function start({ name: rawName, email: rawEmail, password, birthday }) {
  const name = checkName(rawName);
  const email = normEmail(rawEmail);
  checkPassword(password, name, email);
  checkAge(birthday);                         // UNDER_AGE -> nothing is stored
  if (await emailTaken(email)) {
    throw new SignupError(409, "EMAIL_EXISTS", "There's already an account with this email.");
  }

  // tidy up: expired rows, and any earlier attempt with this email (starting again)
  await supabase.from("signup_pending").delete().lt("expires_at", now().toISOString());
  await supabase.from("signup_pending").delete().eq("email", email);

  const id = crypto.randomUUID();
  const code = newCode();
  const { error } = await supabase.from("signup_pending").insert({
    id, email, full_name: name, birthday,
    password_hash: hashPassword(password),
    code_hash: hashCode(id, code),
    resend_after: later(RESEND_AFTER_S),
    code_expires_at: later(CODE_TTL_S),
    expires_at: later(CODE_TTL_S),
  });
  if (error) {
    console.error("[signup] pending insert failed:", error.message);
    throw new SignupError(503, "DB_UNAVAILABLE", "We couldn't reach the server just now. Please try again.");
  }
  if (!(await sendCode(email, name, code))) {
    await supabase.from("signup_pending").delete().eq("id", id);
    throw new SignupError(503, "EMAIL_UNAVAILABLE", "We couldn't send your code just now. Please try again.");
  }
  return { pending_id: id, resend_after: RESEND_AFTER_S, expires_in: CODE_TTL_S };
}

export async function resend({ pending_id }) {
  const row = await getPending(pending_id);
  if (!row) throw notFound();
  if (row.verified_at) throw new SignupError(400, "ALREADY_VERIFIED", "Your email is already confirmed.");
  if (row.attempts >= MAX_ATTEMPTS) {
    throw new SignupError(429, "CODE_LOCKED", "Too many wrong codes. Please start again.");
  }
  if (new Date(row.resend_after) > now()) {
    const wait = secondsUntil(row.resend_after);
    throw new SignupError(429, "RESEND_TOO_SOON", `You can ask for a new code in ${wait} seconds.`,
      { retry_after: wait });
  }
  if (row.resends >= MAX_RESENDS) {
    throw new SignupError(429, "RESEND_LIMIT", "Too many codes sent. Please start again.");
  }
  const code = newCode();
  await supabase.from("signup_pending").update({
    code_hash: hashCode(row.id, code),
    code_expires_at: later(CODE_TTL_S),
    expires_at: later(CODE_TTL_S),
    resend_after: later(RESEND_AFTER_S),
    resends: row.resends + 1,
  }).eq("id", row.id);
  if (!(await sendCode(row.email, row.full_name, code))) {
    await supabase.from("signup_pending").update({ resend_after: now().toISOString() }).eq("id", row.id);
    throw new SignupError(503, "EMAIL_UNAVAILABLE", "We couldn't send your code just now. Please try again.");
  }
  return { resend_after: RESEND_AFTER_S, expires_in: CODE_TTL_S };
}

export async function verify({ pending_id, code }) {
  const row = await getPending(pending_id);
  if (!row) throw notFound();
  if (row.verified_at) return { verified: true };
  if (row.attempts >= MAX_ATTEMPTS) {
    throw new SignupError(429, "CODE_LOCKED", "Too many wrong codes. Please start again.");
  }
  if (new Date(row.code_expires_at) <= now()) {
    throw new SignupError(410, "CODE_EXPIRED", "This code has expired. Ask for a new one.");
  }
  const given = String(code ?? "").trim();
  if (/^\d{6}$/.test(given) && sameHex(hashCode(row.id, given), row.code_hash)) {
    await supabase.from("signup_pending").update({
      verified_at: now().toISOString(), code_hash: null, expires_at: later(VERIFIED_TTL_S),
    }).eq("id", row.id);
    return { verified: true };
  }
  // wrong: count it (only if nobody else counted at the same moment)
  const attempts = row.attempts + 1;
  await supabase.from("signup_pending").update({ attempts }).eq("id", row.id).eq("attempts", row.attempts);
  if (attempts >= MAX_ATTEMPTS) {
    throw new SignupError(429, "CODE_LOCKED", "Too many wrong codes. Please start again.");
  }
  throw new SignupError(400, "CODE_WRONG", "That code isn't right.", { attempts_left: MAX_ATTEMPTS - attempts });
}

export async function complete({ pending_id, terms_version, password }) {
  const terms = String(terms_version ?? "").trim();
  if (!terms || terms.length > 20) {
    throw new SignupError(400, "TERMS_REQUIRED", "Please accept the terms to continue.");
  }
  const row = await getPending(pending_id);
  if (!row) throw notFound();
  if (!row.verified_at) throw new SignupError(400, "NOT_VERIFIED", "Please confirm your email first.");
  if (!passwordMatches(password, row.password_hash)) {
    throw new SignupError(400, "PASSWORD_MISMATCH", "That isn't the password you chose. Please start again.");
  }
  if (await emailTaken(row.email)) {
    await supabase.from("signup_pending").delete().eq("id", row.id);
    throw new SignupError(409, "EMAIL_EXISTS", "There's already an account with this email.");
  }

  // 1. login account (already confirmed: we just checked the email with our code)
  const { data: created, error: createError } = await supabase.auth.admin.createUser({
    email: row.email, password, email_confirm: true, user_metadata: { full_name: row.full_name },
  });
  if (createError || !created?.user) {
    const msg = (createError?.message || "").toLowerCase();
    console.error("[signup] createUser failed for", row.email, ":", createError?.message);
    if (msg.includes("already")) {
      throw new SignupError(409, "EMAIL_EXISTS", "There's already an account with this email.");
    }
    if (msg.includes("password")) {
      throw new SignupError(400, "PASSWORD_WEAK", "Please choose a stronger password.");
    }
    throw new SignupError(500, "SIGNUP_FAILED", "Something went wrong creating your account. Please try again.");
  }
  const user = created.user;
  const rollback = async () => {
    await supabase.from("profiles").delete().eq("id", user.id);
    const { error } = await supabase.auth.admin.deleteUser(user.id);
    if (error) console.error("[signup] rollback of auth user", user.id, "failed:", error.message);
  };

  // 2. profile (setup_status defaults to in_progress / step handle)
  const { error: profileError } = await supabase.from("profiles").insert({
    id: user.id, full_name: row.full_name, email: row.email, tier: "free",
  });
  if (profileError) {
    console.error("[signup] profiles insert failed for", row.email, ":", profileError.message);
    await rollback();
    if ((profileError.message || "").includes("duplicate")) {
      throw new SignupError(409, "EMAIL_EXISTS", "There's already an account with this email.");
    }
    throw new SignupError(500, "SIGNUP_FAILED", "Something went wrong creating your account. Please try again.");
  }

  // 3. private details (never in any public response)
  const { error: privateError } = await supabase.from("user_private").insert({
    user_id: user.id, birthday: row.birthday, terms_version: terms, terms_accepted_at: now().toISOString(),
  });
  if (privateError) {
    console.error("[signup] user_private insert failed for", row.email, ":", privateError.message);
    await rollback();
    throw new SignupError(500, "SIGNUP_FAILED", "Something went wrong creating your account. Please try again.");
  }

  await supabase.from("signup_pending").delete().eq("id", row.id);

  // 4. sign in on a separate client, so the shared service client keeps its service key
  const temp = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  const { data: session, error: signInError } = await temp.auth.signInWithPassword({ email: row.email, password });
  if (signInError || !session?.session) {
    console.error("[signup] sign-in after create failed for", row.email, ":", signInError?.message);
    throw new SignupError(500, "SIGNIN_FAILED", "Your account is ready. Please log in.");
  }

  await sendWelcomeEmail(row.email, row.full_name);       // best-effort, never throws
  console.log("[signup] account created", user.id);
  return {
    message: "Account created",
    user: { id: user.id, email: user.email, full_name: row.full_name },
    access_token: session.session.access_token,
    refresh_token: session.session.refresh_token,
    is_new_user: true,
  };
}