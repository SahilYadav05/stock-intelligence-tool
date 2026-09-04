import { getChatGPTUser } from "@/app/chatgpt-auth";
import { LiveMarketTerminal } from "@/src/components/live-market-terminal";

export const dynamic = "force-dynamic";

export default async function Home() {
  const user = await getChatGPTUser();
  return <LiveMarketTerminal authenticated={Boolean(user)} />;
}
