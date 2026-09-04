import {
  isMarketStateView,
  parseMarketSocketMessage,
  type DataStatus,
  type MarketStateViewContract,
} from "@/src/lib/market-contracts";

export interface MarketStreamCallbacks {
  onConnecting: () => void;
  onStatus: (status: DataStatus, reason: string) => void;
  onView: (view: MarketStateViewContract) => void;
}

export function connectMarketStream(
  apiBaseUrl: string,
  callbacks: MarketStreamCallbacks,
): () => void {
  let stopped = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;
  const normalizedBase = apiBaseUrl.replace(/\/$/, "");

  const fetchCurrent = async () => {
    try {
      const response = await fetch(
        `${normalizedBase}/api/v1/market-state/NIFTY50_SPOT?timeframe=5m`,
        { headers: { Accept: "application/json" } },
      );
      if (response.ok) {
        const body: unknown = await response.json();
        if (isMarketStateView(body)) callbacks.onView(body);
        return;
      }
      callbacks.onStatus("DISCONNECTED", await responseReason(response));
    } catch {
      callbacks.onStatus("DISCONNECTED", "MARKET_API_UNREACHABLE");
    }
  };

  const openSocket = () => {
    if (stopped) return;
    callbacks.onConnecting();
    // A REST request provides a useful fail-closed reason even if a gateway
    // is intentionally not configured to accept WebSocket upgrades yet.
    void fetchCurrent();
    let socketUrl: URL;
    try {
      socketUrl = new URL(normalizedBase);
    } catch {
      callbacks.onStatus("DISCONNECTED", "INVALID_MARKET_API_URL");
      return;
    }
    socketUrl.protocol = socketUrl.protocol === "https:" ? "wss:" : "ws:";
    socketUrl.pathname = "/ws/v1/market-state";
    socketUrl.search = new URLSearchParams({
      instrument_id: "NIFTY50_SPOT",
      timeframe: "5m",
    }).toString();

    socket = new WebSocket(socketUrl);
    socket.onopen = () => { attempt = 0; };
    socket.onmessage = (event) => {
      try {
        const message = parseMarketSocketMessage(JSON.parse(String(event.data)));
        if (!message) {
          callbacks.onStatus("STALE", "INVALID_MARKET_MESSAGE");
          return;
        }
        if (message.message_type === "STATUS") {
          callbacks.onStatus(message.payload.data_status, message.payload.reason);
        } else {
          callbacks.onView(message.payload);
        }
      } catch {
        callbacks.onStatus("STALE", "INVALID_MARKET_MESSAGE");
      }
    };
    socket.onerror = () => callbacks.onStatus("DISCONNECTED", "MARKET_API_UNREACHABLE");
    socket.onclose = () => {
      if (stopped) return;
      callbacks.onStatus("DISCONNECTED", "MARKET_STREAM_CLOSED");
      const delay = Math.min(1_000 * 2 ** attempt, 15_000);
      attempt += 1;
      reconnectTimer = setTimeout(openSocket, delay);
    };
  };

  openSocket();
  return () => {
    stopped = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    socket?.close(1000, "client shutdown");
  };
}

async function responseReason(response: Response): Promise<string> {
  try {
    const value: unknown = await response.json();
    if (typeof value === "object" && value !== null && "reason" in value) {
      const reason = (value as { reason?: unknown }).reason;
      if (typeof reason === "string") return reason;
    }
  } catch {
    // A non-JSON upstream error must not be trusted as UI copy.
  }
  return `MARKET_API_HTTP_${response.status}`;
}
