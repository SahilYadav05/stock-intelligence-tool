import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NIFTY Intelligence Terminal",
  description:
    "A private, evidence-first quantitative research and trading-intelligence terminal.",
  openGraph: {
    title: "NIFTY Intelligence Terminal",
    description: "Private quantitative research terminal",
    images: [{ url: "/og.png", width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "NIFTY Intelligence Terminal",
    description: "Private quantitative research terminal",
    images: ["/og.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
