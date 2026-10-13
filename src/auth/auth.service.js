import redis from "../config/redis.js";
import supabase from "../config/supabase.js";
import { 
  sendWelcomeEmail,
  sendLoginAlertEmail,
  sendAccountDeletedEmail
} from '../config/emailService.js'

export const signUpUser = async ({ email, password, fullName }) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: { full_name: fullName }
    }
  })

  if (error) {
    // Intercept Supabase auth errors
    if (error.message.includes('already registered')) {
      throw new Error('An account with this email already exists. Please log in instead.')
    }
    if (error.message.includes('invalid email')) {
      throw new Error('Please enter a valid email address.')
    }
    if (error.message.includes('weak password')) {
      throw new Error('Your password is too weak. Use at least 8 characters with a number and uppercase letter.')
    }
    throw new Error('Something went wrong creating your account. Please try again.')
  }

  const { error: profileError } = await supabase
    .from('profiles')
    .insert({
      id: data.user.id,
      full_name: fullName,
      email,
      tier: 'free'
    })

  if (profileError) {
    // Intercept database errors
    if (profileError.message.includes('duplicate key') || 
        profileError.message.includes('unique constraint')) {
      throw new Error('An account with this email already exists. Please log in instead.')
    }
    if (profileError.message.includes('null value')) {
      throw new Error('Please fill in all required fields.')
    }
    throw new Error('Something went wrong setting up your profile. Please try again.')
  }

    await sendWelcomeEmail(email, fullName)  // after all checks pass
    return data.user
}

export const loginUser = async ({ email, password }) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  })

  if (error) {
    if (error.message.includes('Invalid login credentials')) {
      throw new Error('Incorrect email or password. Please try again.')
    }
    if (error.message.includes('Email not confirmed')) {
      throw new Error('Please confirm your email before logging in. Check your inbox.')
    }
    if (error.message.includes('too many requests')) {
      throw new Error('Too many login attempts. Please wait a few minutes and try again.')
    }
    throw new Error('Something went wrong logging in. Please try again.')
  }
    const time = new Date().toLocaleString('en-US', { 
    dateStyle: 'medium', 
    timeStyle: 'short' 
    })
    const device = 'Unknown device' 
const { data: profile } = await supabase
    .from('profiles')
    .select('full_name')
    .eq('id', data.user.id)
    .single()

await sendLoginAlertEmail(data.user.email, profile?.full_name, time, device)
  return {
    user: data.user,
    access_token: data.session.access_token,
    refresh_token: data.session.refresh_token
  }
}

export const refreshToken = async (refresh_token) => {
    const {data, error} = await supabase.auth.refreshSession({
        refresh_token
    })
  if (error) throw new Error('Session expired, please login again');

  return{
    access_token: data.session.access_token,
    refresh_token: data.session.refresh_token
  }
}

export const forgotPassword = async (email) =>{
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: 'fitcheck://reset-password'
    });
    if (error) throw new Error(error.message);
}

export const resetPassword = async (password) => {
    const {error} = await supabase.auth.updateUser({
        password
    })
    if(error) throw new Error(error.message);
}

export const logoutUser = async (token) => {
    await supabase.auth.signOut()
    await redis.del(`session:${token}`)
}

export const getProfile = async (userId) => {
    const {data, error} = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single()

    if(error) throw new Error(error.message);

    return data
}

export const deleteProfile = async (userId, token) => {
    const {error} = await supabase
    .from('profiles')
    .delete()
    .eq('id', userId)

    if(error) throw new Error(error.message);

    await supabase.auth.admin.deleteUser(userId);
    
    const { data: profile } = await supabase
    .from('profiles')
    .select('full_name, email')
    .eq('id', userId)
    .single()

    await sendAccountDeletedEmail(profile.email, profile.full_name)
    await supabase.from('profiles').delete().eq('id', userId)
    await supabase.auth.admin.deleteUser(userId)
    await redis.del(`session:${token}`)
  }

  export const googleLogin = async (idToken) => {
    const jwtRegex = /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$/
    if (!jwtRegex.test(idToken)) {
      throw new Error('Invalid token format. Please try again.')
    }

    const {data, error} = await supabase.auth.signInWithIdToken({
      provider: 'google',
      token: idToken
    })

  if(error){
    if(error.message.includes('invalid')){
      throw new Error('Google sign in failed. Please try again.')
    }
    if(error.message.includes('already exists') || 
      error.message.includes('already registered')){
      throw new Error('An account with this email already exists. Please log in with your email and password instead.')
    }
    throw new Error('Something went wrong with Google login. Please try again.')
  }
  
  const {data : existingProfile} =  await supabase
  .from('profiles')
  .select('id')
  .eq('id', data.user.id)
  .single()

  if(!existingProfile){
    const {error : profileError} = await supabase
    .from('profiles')
    .insert({
       id: data.user.id,
        full_name: data.user.user_metadata.full_name,
        email: data.user.email,
        tier: 'free',
        auth_provider: 'google'
    })

    if(profileError) {
      throw new Error('Something went wrong setting up your account. Please try again.')
    }

    await sendWelcomeEmail(data.user.email, data.user.user_metadata.full_name)

  }

  return {
    user: data.user,
    access_token: data.session.access_token,
    refresh_token: data.session.refresh_token,
    isNewUser: !existingProfile 
  }

}

export const verifyResetOtp = async (email, token) => {
  const { data, error } = await supabase.auth.verifyOtp({
    email,
    token,
    type: 'recovery'
  })
  if (error) throw new Error('Invalid or expired code. Please try again.')
  return {
    access_token: data.session.access_token,
    refresh_token: data.session.refresh_token
  }
}

const authService = {
    signUpUser,
    loginUser,
    refreshToken,
    forgotPassword,
    resetPassword,
    logoutUser,
    getProfile,
    deleteProfile,
    googleLogin,
    verifyResetOtp
}
export default authService; 