// src/config/emailService.js
//
// Transactional email via Brevo's HTTP API (no SDK needed).
// Brevo lets you send from a single VERIFIED sender address (e.g. your
// Gmail) without owning a domain.
//
// .env:
//   BREVO_API_KEY=xkeysib-...
//   EMAIL_FROM=the.address.you.verified.in.brevo@gmail.com
//   EMAIL_FROM_NAME=FitCheck
//
// RULE: email is best-effort. These functions start the send and return
// immediately. A failed or slow email is logged and NEVER fails or slows
// signup / login / account deletion.

import {
  welcomeEmail,
  passwordResetEmail,
  loginAlertEmail,
  accountDeletedEmail,
} from "./emailTemplates.js";

const BREVO_URL = "https://api.brevo.com/v3/smtp/email";
const SEND_TIMEOUT_MS = 10000;

function deliver(kind, to, template) {
  const apiKey = process.env.BREVO_API_KEY;
  const fromEmail = process.env.EMAIL_FROM;
  const fromName = process.env.EMAIL_FROM_NAME || "FitCheck";

  if (!apiKey || !fromEmail) {
    console.warn(`[email] ${kind} not sent: BREVO_API_KEY or EMAIL_FROM missing in .env`);
    return;
  }
  if (!to) {
    console.warn(`[email] ${kind} not sent: no recipient`);
    return;
  }

  fetch(BREVO_URL, {
    method: "POST",
    headers: {
      "api-key": apiKey,
      "content-type": "application/json",
      accept: "application/json",
    },
    body: JSON.stringify({
      sender: { email: fromEmail, name: fromName },
      to: [{ email: to }],
      subject: template.subject,
      htmlContent: template.html,
    }),
    signal: AbortSignal.timeout(SEND_TIMEOUT_MS),
  })
    .then(async (res) => {
      if (res.ok) {
        console.log(`[email] ${kind} sent to ${to}`);
      } else {
        const body = await res.text().catch(() => "");
        console.error(`[email] ${kind} to ${to} failed: HTTP ${res.status} ${body}`);
      }
    })
    .catch((err) => {
      console.error(`[email] ${kind} to ${to} failed: ${err.message}`);
    });
}

export const sendWelcomeEmail = async (email, fullName) => {
  deliver("welcome", email, welcomeEmail(fullName));
};

export const sendPasswordResetEmail = async (email, resetLink) => {
  deliver("password-reset", email, passwordResetEmail(resetLink));
};

export const sendLoginAlertEmail = async (email, fullName, time, device) => {
  deliver("login-alert", email, loginAlertEmail(fullName, time, device));
};

export const sendAccountDeletedEmail = async (email, fullName) => {
  deliver("account-deleted", email, accountDeletedEmail(fullName));
};