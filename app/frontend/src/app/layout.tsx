import type { Metadata } from "next";
import { Irish_Grover, Poppins } from "next/font/google";
import "./globals.css";

const irishGrover = Irish_Grover({
  variable: "--font-irish-grover",
  subsets: ["latin"],
  weight: "400", 
});

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"], 
});

export const metadata: Metadata = {
  title: "FeelOut",
  description: "Analyzes text files to detect and classify emotions",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${irishGrover.variable} ${poppins.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}