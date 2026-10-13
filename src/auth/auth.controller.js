import authService, {
  logoutUser,
  signUpUser,
  refreshToken,
  forgotPassword as forgotPasswordService,
  resetPassword as resetPasswordService,
  deleteProfile,
} from "./auth.service.js";
import authValidation, { validateSignUp, validateResetPassword } from "./auth.validation.js";
import { sendUnavailable } from "../middleware/authenticate.js";

export const signUp = async (req, res) => {
  try {
    const errors = validateSignUp(req.body);
    if (Object.keys(errors).length > 0) {
      return res.status(400).json({ error: 'Validation failed', errors });
    }

    const { email, fullName, password } = req.body;
    const user = await signUpUser({ email, fullName, password });

    res.status(201).json({
      message: 'Account created successfully, please check your email to verify your account',
      userId: user.id,
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
}

export const login = async (req, res) => {
  try {
    const errors = authValidation.validateLogin(req.body);
    if (Object.keys(errors).length > 0) {
      return res.status(400).json({ error: 'Validation failed', errors });
    }

    const { email, password } = req.body;
    const result = await authService.loginUser({ email, password });

    res.json({
      message: 'Login successful',
      user: {
        id: result.user.id,
        email: result.user.email,
      },
      access_token: result.access_token,
      refresh_token: result.refresh_token,
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const refresh = async (req, res) => {
  try {
    const { refresh_token } = req.body;

    if (!refresh_token) {
      return res.status(400).json({ error: 'Refresh token is required' });
    }

    const tokens = await refreshToken(refresh_token);
    res.json({
      message: 'Token refreshed successfully',
      tokens: tokens,
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;

    if (!email || !email.includes('@')) {
      return res.status(400).json({ error: 'Invalid email address' });
    }

    await forgotPasswordService(email);
    res.json({ message: 'Password reset email sent, please check your inbox' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const resetPassword = async (req, res) => {
  try {
    const errors = validateResetPassword(req.body);
    if (Object.keys(errors).length > 0) {
      return res.status(400).json({ error: 'Validation failed', errors });
    }

    const { password } = req.body;
    await resetPasswordService(password);

    res.json({ message: 'Password reset successful, you can now login with your new password' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const logout = async (req, res) => {
  try {
    const token = req.headers.authorization?.split('Bearer ')[1];
    await logoutUser(token);
    res.json({ message: 'Logout successful' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Profile is already loaded (and cached) by the authenticate middleware,
// so /auth/me makes no extra database call.
//   503 PROFILE_UNAVAILABLE -> database/network problem, retry (do NOT log out)
//   404 PROFILE_NOT_FOUND   -> valid user, no profiles row
export const getMe = async (req, res) => {
  if (req.profileUnavailable) {
    return sendUnavailable(res, 'PROFILE_UNAVAILABLE', 'Profile temporarily unavailable, please retry');
  }
  if (!req.profile) {
    return res.status(404).json({ error: 'Profile not found', code: 'PROFILE_NOT_FOUND' });
  }
  res.json({ user: req.profile });
};

export const deleteAccount = async (req, res) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    await deleteProfile(req.user.id, token);
    res.json({ message: 'Account deleted successfully' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

export const googleLogin = async (req, res) => {
  try {
    const { idToken } = req.body;

    if (!idToken) {
      return res.status(400).json({ error: 'Google token is required' });
    }
    const result = await authService.googleLogin(idToken);

    res.json({
      message: result.isNewUser ? 'Account created successfully' : 'Login successful',
      user: {
        id: result.user.id,
        email: result.user.email,
        full_name: result.user.user_metadata.full_name,
      },
      access_token: result.access_token,
      refresh_token: result.refresh_token,
      isNewUser: result.isNewUser,
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// auth.controller.js
export const verifyOtp = async (req, res) => {
  try {
    const { email, token } = req.body
    const tokens = await authService.verifyResetOtp(email, token)
    res.json({ message: 'Code verified', ...tokens })
  } catch (error) {
    res.status(400).json({ error: error.message })
  }
}

const authController = {
  signUp,
  login,
  googleLogin,
  refresh,
  forgotPassword,
  resetPassword,
  logout,
  getMe,
  deleteAccount,
  verifyOtp
};
export default authController;