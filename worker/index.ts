/** Cloudflare Worker entry point for the vinext-starter template. */
import { handleImageOptimization, DEFAULT_DEVICE_SIZES, DEFAULT_IMAGE_SIZES } from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";

interface Env {
  ASSETS: Fetcher;
  DB: D1Database;
  /**
   * Private HTTPS origin for the persistent Python market service. This is a
   * Worker secret/configuration value, never a NEXT_PUBLIC browser value.
   */
  MARKET_API_BASE_URL?: string;
  /** Server-to-server bearer credential for the market service. */
  MARKET_API_BEARER_TOKEN?: string;
  IMAGES: {
    input(stream: ReadableStream): {
      transform(options: Record<string, unknown>): {
        output(options: { format: string; quality: number }): Promise<{ response(): Response }>;
      };
    };
  };
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

// Image security config. SVG sources with .svg extension auto-skip the
// optimization endpoint on the client side (served directly, no proxy).
// To route SVGs through the optimizer (with security headers), set
// dangerouslyAllowSVG: true in next.config.js and uncomment below:
// const imageConfig: ImageConfig = { dangerouslyAllowSVG: true };

const worker = {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (isMarketGatewayPath(url.pathname)) {
      return forwardMarketRequest(request, env);
    }

    if (url.pathname === "/_vinext/image") {
      const allowedWidths = [...DEFAULT_DEVICE_SIZES, ...DEFAULT_IMAGE_SIZES];
      return handleImageOptimization(request, {
        fetchAsset: (path) => env.ASSETS.fetch(new Request(new URL(path, request.url))),
        transformImage: async (body, { width, format, quality }) => {
          const result = await env.IMAGES.input(body).transform(width > 0 ? { width } : {}).output({ format, quality });
          return result.response();
        },
      }, allowedWidths);
    }

    return handler.fetch(request, env, ctx);
  },
};

export default worker;

const MARKET_GATEWAY_PREFIXES = ["/api/v1/", "/ws/v1/"] as const;

function isMarketGatewayPath(pathname: string): boolean {
  return MARKET_GATEWAY_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

/**
 * The Worker is deliberately a narrow BFF, not a general-purpose proxy. It
 * provides an authenticated same-origin route for the private dashboard while
 * keeping the provider, ledger, inference process and bearer credential off
 * the browser. The Python service remains the authority for snapshot identity
 * and price-action calculations.
 */
async function forwardMarketRequest(request: Request, env: Env): Promise<Response> {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return gatewayResponse(405, "MARKET_GATEWAY_READ_ONLY");
  }

  const baseUrl = parsePrivateApiUrl(env.MARKET_API_BASE_URL);
  const token = env.MARKET_API_BEARER_TOKEN?.trim();
  if (!baseUrl || !token) {
    return gatewayResponse(503, "MARKET_API_PROXY_NOT_CONFIGURED");
  }

  const incomingUrl = new URL(request.url);
  const upstreamUrl = new URL(`${incomingUrl.pathname}${incomingUrl.search}`, baseUrl.origin);
  const headers = new Headers(request.headers);
  // Browser-supplied credentials must never select the upstream identity.
  headers.delete("authorization");
  headers.delete("cookie");
  headers.delete("host");
  headers.set("authorization", `Bearer ${token}`);
  headers.set("cache-control", "no-store");
  headers.set("x-terminal-gateway", "cloudflare-worker");

  try {
    const upstream = await fetch(upstreamUrl, {
      method: request.method,
      headers,
      redirect: "manual",
    });

    // A WebSocket upgrade must retain the upstream response unchanged.
    if (request.headers.get("upgrade")?.toLowerCase() === "websocket") {
      return upstream;
    }

    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.set("cache-control", "no-store");
    responseHeaders.set("x-content-type-options", "nosniff");
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch {
    // Do not reveal the private upstream origin or network details.
    return gatewayResponse(502, "MARKET_API_UNREACHABLE");
  }
}

function parsePrivateApiUrl(value: string | undefined): URL | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url : null;
  } catch {
    return null;
  }
}

function gatewayResponse(status: number, reason: string): Response {
  return Response.json(
    { schema_version: 1, data_status: "DISCONNECTED", reason },
    {
      status,
      headers: {
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
      },
    },
  );
}
