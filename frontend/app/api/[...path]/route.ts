import { NextRequest } from "next/server";

// Proxies /api/* to the FastAPI backend inside the Docker network, so the
// browser only ever talks to the frontend origin (no CORS, one exposed port).
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";
const PASS_HEADERS = ["content-type", "cache-control", "content-disposition"];

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const target = `${BACKEND_URL}/${path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;

  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  const init: RequestInit & { duplex?: "half" } = { method: req.method, headers, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body; // stream uploads straight through
    init.duplex = "half";
  }

  let res: Response;
  try {
    res = await fetch(target, init);
  } catch {
    return Response.json({ detail: "Backend is unreachable. Is the backend container running?" }, { status: 502 });
  }

  const out = new Headers();
  for (const h of PASS_HEADERS) {
    const v = res.headers.get(h);
    if (v) out.set(h, v);
  }
  out.set("x-accel-buffering", "no");
  const body = res.status === 204 || res.status === 304 ? null : res.body;
  return new Response(body, { status: res.status, headers: out });
}

export { proxy as GET, proxy as POST, proxy as DELETE };
