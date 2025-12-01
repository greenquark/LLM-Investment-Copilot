import { StdioServerTransport } from "@modelcontextprotocol/sdk/transport/server";
import { Server } from "@modelcontextprotocol/sdk/server";
import { z } from "zod";
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

const server = new Server(
  {
    name: "trading-agent-mcp",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.tool("backtest_wheel", {
  description: "Run Wheel backtest via MarketData.app using config/env.backtest.yaml",
  inputSchema: z.object({
    symbol: z.string().optional(),
    start: z.string().optional(),
    end: z.string().optional(),
    initialCash: z.number().optional(),
  }),
  async execute({ input }) {
    const { stdout, stderr } = await execFileAsync("python", [
      "scripts/run_backtest_wheel.py",
    ]);
    return {
      content: [
        {
          type: "text",
          text: stdout || stderr || "No output",
        },
      ],
    };
  },
});

server.tool("test_wheel_performance", {
  description:
    "Run Wheel backtest and compare against buy-and-hold using config/env.backtest.yaml",
  inputSchema: z.object({}),
  async execute() {
    const { stdout, stderr } = await execFileAsync("python", [
      "scripts/test_wheel_performance.py",
    ]);
    return {
      content: [
        {
          type: "text",
          text: stdout || stderr || "No output",
        },
      ],
    };
  },
});

server.tool("live_status", {
  description: "Stub for live engine status (extend to query your live process).",
  inputSchema: z.object({}),
  async execute() {
    return {
      content: [
        {
          type: "text",
          text: "Live engine status: stub. Wire this to your running live engine or monitoring endpoint.",
        },
      ],
    };
  },
});

const transport = new StdioServerTransport();
server.connect(transport);
