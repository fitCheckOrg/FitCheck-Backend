import express from "express";
import authController from "./auth.controller.js";
import rateLimit from "express-rate-limit";
import { authenticate } from "../middleware/authenticate.js";

const authRouter = express.Router();

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: { error: 'Too many attempts. Please wait a few minutes and try again.' },
});

authRouter.post("/signup", authLimiter, authController.signUp);
authRouter.post("/login", authLimiter, authController.login);
authRouter.post("/refresh", authController.refresh);
authRouter.post("/forgot-password", authLimiter, authController.forgotPassword);
authRouter.post("/reset-password", authController.resetPassword);
// auth.routes.js
authRouter.post("/verify-reset-otp", authLimiter, authController.verifyOtp);
authRouter.post("/google", authLimiter, authController.googleLogin);

// protected routes 
authRouter.get("/me", authenticate, authController.getMe);
authRouter.post("/logout", authenticate, authController.logout);
authRouter.delete("/account", authenticate, authController.deleteAccount);

export default authRouter;