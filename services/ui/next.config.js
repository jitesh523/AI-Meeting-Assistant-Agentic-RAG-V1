/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
  },
  async rewrites() {
    return [
      {
        source: '/api/ingestion/:path*',
        destination: 'http://ingestion:8000/:path*',
      },
      {
        source: '/api/asr/:path*',
        destination: 'http://asr:8000/:path*',
      },
      {
        source: '/api/nlu/:path*',
        destination: 'http://nlu:8000/:path*',
      },
      {
        source: '/api/rag/:path*',
        destination: 'http://rag:8000/:path*',
      },
      {
        source: '/api/agent/:path*',
        destination: 'http://agent:8000/agent/:path*',
      },
      {
        source: '/api/integrations/:path*',
        destination: 'http://integrations:8000/:path*',
      },
    ]
  },
}

module.exports = nextConfig
