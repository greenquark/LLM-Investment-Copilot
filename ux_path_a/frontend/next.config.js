/** @type {import('next').NextConfig} */
const pkg = require('./package.json')

const nextConfig = {
  reactStrictMode: true,
  // Expose build metadata to the client for UI display (safe: commit sha/ref/version only)
  env: {
    NEXT_PUBLIC_APP_VERSION: pkg.version,
    NEXT_PUBLIC_BUILD_SHA:
      process.env.VERCEL_GIT_COMMIT_SHA ||
      process.env.GITHUB_SHA ||
      process.env.RAILWAY_GIT_COMMIT_SHA ||
      '',
    NEXT_PUBLIC_BUILD_REF:
      process.env.VERCEL_GIT_COMMIT_REF ||
      process.env.GITHUB_REF_NAME ||
      process.env.RAILWAY_GIT_BRANCH ||
      '',
    NEXT_PUBLIC_BUILD_TIME: new Date().toISOString(),
  },
}

module.exports = nextConfig
