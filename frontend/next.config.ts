import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // gzip would buffer the NDJSON token stream coming from the backend
  compress: false,
  poweredByHeader: false,
};

export default nextConfig;
