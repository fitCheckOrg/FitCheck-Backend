import redis from "../config/redis.js";
import supabase from "../config/supabase.js";
import {
  sendWelcomeEmail,
  sendLoginAlertEmail,
  sendAccountDeletedEmail
} from '../config/emailService.js'

// A profiles row whose auth user no longer exists (e.g. the user was deleted
// from the Supabase dashboard) blocks anyone signing up again with that email.
// Returns true if such an orphan was found and removed.
const removeOrphanProfile = async (email, newUserId) => {
  const { data: rows, error } = await supabase
    .from('profiles')
    .select('id, email')
    .in('email', [...new Set([email, email.toLowerCase()])])

  if (error || !rows || rows.length === 0) return false

  let removed = false
  for (const row of rows) {
    if (row.id === newUserId) continue
    const { data: owner, error: ownerError } = await supabase.auth.admin.getUserById(row.id)
    const notFound = !owner?.user && (!ownerError || ownerError.status === 404 ||
      (ownerError.message || '').toLowerCase().includes('not found'))
    if (!notFound) continue   // a real account owns it, or we couldn't check - leave it

    const { error: delError } = await supabase.from('profiles').delete().eq('id', row.id)
    if (delError) {
      console.error('[signup] could not remove orphan profile', row.id, delError.message)
      continue
    }
    console.warn(`[signup] removed orphan profile ${row.id} (${row.email}) - its auth user no longer exists`)
    removed = true
  }
  return removed
}

// If the profile can't be created, delete the auth user we just made, so the
// person isn't left with a half-created account they can't sign up OR log in with.
const rollbackAuthUser = async (userId) => {
  const { error } = await supabase.auth.admin.deleteUser(userId)
  if (error) console.error('[signup] rollback of auth user', userId, 'failed:', error.message)
  else console.warn('[signup] rolled back auth user', userId)
}

export const signUpUser = async ({ email, password, fullName }) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: { full_name: fullName }
    }
  })

  if (error) {
    // Always log the REAL Supabase error - the user only sees the friendly one.
    console.error('[signup] supabase.auth.signUp failed for', email, ':', error.status, error.code, error.message)

    const msg = (error.message || '').toLowerCase()
    if (msg.includes('already registered')) {
      throw new Error('An account with this email already exists. Please log in instead.')
    }
    if (msg.includes('invalid email') || msg.includes('is invalid')) {
      throw new Error('Please enter a valid email address.')
    }
    if (msg.includes('weak password') || msg.includes('password should')) {
      throw new Error('Your password is too weak. Use at least 8 characters with a number and uppercase letter.')
    }
    if (msg.includes('rate limit') || error.status === 429) {
      throw new Error('Too many sign-up attempts right now. Please wait a few minutes and try again.')
    }
    throw new Error('Something went wrong creating your account. Please try again.')
  }

  if (!data?.user) {
    console.error('[signup] supabase.auth.signUp returned no user for', email)
    throw new Error('Something went wrong creating your account. Please try again.')
  }

  // With email confirmation on, Supabase does NOT error for an email that is
  // already registered - it returns a placeholder user with no identities.
  if (Array.isArray(data.user.identities) && data.user.identities.length === 0) {
    console.warn('[signup] email already registered:', email)
    throw new Error('An account with this email already exists. Please log in instead.')
  }

  const profileRow = {
    id: data.user.id,
    full_name: fullName,
    email: data.user.email || email,
    tier: 'free'
  }

  let { error: profileError } = await supabase.from('profiles').insert(profileRow)

  if (profileError && profileError.code === '23505' &&
      (profileError.message || '').includes('email')) {
    const cleaned = await removeOrphanProfile(email, data.user.id)
    if (cleaned) {
      ({ error: profileError } = await supabase.from('profiles').insert(profileRow))
    }
  }

  if (profileError) {
    console.error('[signup] profiles insert failed for', email, ':', profileError.code, profileError.message)
    await rollbackAuthUser(data.user.id)

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

    await sendWelcomeEmail(email, fullName)  // best-effort, never throws
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
    // 1. Read name/email FIRST - needed for the goodbye email after deletion.
    const { data: profile } = await supabase
    .from('profiles')
    .select('full_name, email')
    .eq('id', userId)
    .maybeSingle()

    // 2. Delete the login account. If this fails, nothing has been removed yet.
    const { error: authError } = await supabase.auth.admin.deleteUser(userId)
    if (authError) {
      console.error('[delete-account] auth user delete failed for', userId, ':', authError.message)
      throw new Error('Could not delete your account right now. Please try again.')
    }

    // 3. Delete the profile row. If this fails the row is an orphan, which the
    //    signup flow cleans up automatically - the account itself is gone.
    const { error: profileError } = await supabase
    .from('profiles')
    .delete()
    .eq('id', userId)
    if (profileError) {
      console.error('[delete-account] profile row not removed for', userId, ':', profileError.message)
    }

    // 4. Clear cached session/profile data (best-effort).
    try {
      await redis.del(`session:${token}`)
      await redis.del(`profile:${userId}`)
    } catch (err) {
      console.warn('[delete-account] cache cleanup failed:', err.message)
    }

    // 5. Goodbye email (best-effort, never throws).
    if (profile?.email) {
      await sendAccountDeletedEmail(profile.email, profile.full_name)
    }
    console.log('[delete-account] deleted user', userId)
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