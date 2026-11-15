import express from "express";
import authController from "./auth.controller.js";
import signupController from "./signup.controller.js";
import rateLimit from "express-rate-limit";
import { authenticate } from "../middleware/authenticate.js";

const authRouter = express.Router();

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: { error: 'Too many attempts. Please wait a few minutes and try again.' },
});

// ONB-4: limits for the new sign-up steps (envelope-style answer)
const envelopeLimit = (windowMs, max) => rateLimit({
  windowMs,
  max,
  message: { success: false, error: { code: 'RATE_LIMITED', message: 'Too many attempts. Please wait a few minutes and try again.' } },
});
const emailCheckLimiter = envelopeLimit(60 * 1000, 20);        // 20 a minute per IP
const signupLimiter = envelopeLimit(15 * 60 * 1000, 30);       // 30 per 15 minutes per IP

// old one-shot sign-up (older app versions)
authRouter.post("/signup", authLimiter, authController.signUp);

// ONB-4 A: sign-up with an email code + birthday
authRouter.get("/email-available", emailCheckLimiter, signupController.emailAvailable);
authRouter.post("/signup/start", signupLimiter, signupController.start);
authRouter.post("/signup/resend", signupLimiter, signupController.resend);
authRouter.post("/signup/verify", signupLimiter, signupController.verify);
authRouter.post("/signup/complete", signupLimiter, signupController.complete);

// ONB-4 F2 / F3: login with lockout, forgot password that never reveals an account
authRouter.post("/login", authLimiter, signupController.login);
authRouter.post("/forgot-password", authLimiter, signupController.forgotPassword);

authRouter.post("/refresh", authController.refresh);
authRouter.post("/reset-password", authController.resetPassword);
authRouter.post("/verify-reset-otp", authLimiter, authController.verifyOtp);
authRouter.post("/google", authLimiter, authController.googleLogin);

// protected routes
authRouter.get("/me", authenticate, authController.getMe);
authRouter.post("/logout", authenticate, authController.logout);
authRouter.delete("/account", authenticate, authController.deleteAccount);

export default authRouter;