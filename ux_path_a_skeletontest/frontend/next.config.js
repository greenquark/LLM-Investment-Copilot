/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const backend = process.env.NEXT_PUBLIC_BACKEND_URL
    if (!backend) return []
    return [
      {
        source: '/api/:path*',
        destination: `${backend.replace(/\/$/, '')}/api/:path*`,
      },
    ]
  },
}

module.exports = nextConfig

