import {
  type AgentNamespace,
  type Connection,
  routeAgentRequest,
  type Agent,
  type Schedule,
} from "agents-sdk";
import { AIChatAgent } from "agents-sdk/ai-chat-agent";
import {
  createDataStreamResponse,
  generateId,
  type Message,
  streamText,
  type StreamTextOnFinishCallback,
} from "ai";
import { createOpenAI, openai } from "@ai-sdk/openai";
import { processToolCalls } from "./utils";
import { tools, executions } from "./tools";
import { AsyncLocalStorage } from "node:async_hooks";
import { createDeepSeek } from "@ai-sdk/deepseek";

// Environment variables type definition
export type Env = {
  OPENAI_API_KEY: string;
  DEEPSEEK_API_KEY: string;
  AI_GATEWAY_TOKEN: string;
  Chat: AgentNamespace<Chat>;
};

// we use ALS to expose the agent context to the tools
export const agentContext = new AsyncLocalStorage<Chat>();
/**
 * Chat Agent implementation that handles real-time AI chat interactions
 */
export class Chat extends AIChatAgent<Env> {
  /**
   * Handles incoming chat messages and manages the response stream
   * @param onFinish - Callback function executed when streaming completes
   */
  async onChatMessage(onFinish: StreamTextOnFinishCallback<any>) {
    // Create a streaming response that handles both text and tool outputs
    return agentContext.run(this, async () => {
      const dataStreamResponse = createDataStreamResponse({
        execute: async (dataStream) => {
          console.log("streaming response");
          // Process any pending tool calls from previous messages
          // This handles human-in-the-loop confirmations for tools
          let processedMessages = await processToolCalls({
            messages: this.messages,
            dataStream,
            tools,
            executions,
          });

          // Initialize OpenAI client with API key from environment
          const openai = createOpenAI({
            apiKey: this.env.OPENAI_API_KEY,
            baseURL: "https://gateway.ai.cloudflare.com/v1/f7d1b2da33b7651c618148d0e05d024d/openai-gateway/openai",
            headers: {
              "cf-aig-authorization": `Bearer ${this.env.AI_GATEWAY_TOKEN}`,
            },
          });
          // const deepseek = createDeepSeek({
          //   apiKey: this.env.DEEPSEEK_API_KEY ?? '',
          // });
          
          // Stream the AI response using GPT-4
          const result = streamText({
            model: openai("gpt-4o-mini"),
            system: `
              You are a helpful assistant that can do various tasks. If the user asks, then you can also schedule tasks to be executed later. The input may have a date/time/cron pattern to be input as an object into a scheduler The time is now: ${new Date().toISOString()}.
              `,
            messages: processedMessages,
            tools,
            onFinish,
            maxSteps: 10,
          });

          // Merge the AI response stream with tool execution outputs
          result.mergeIntoDataStream(dataStream);
        },
      });

      return dataStreamResponse;
    });
  }
  async executeTask(description: string, task: Schedule<string>) {
    await this.saveMessages([
      ...this.messages,
      {
        id: generateId(),
        role: "user",
        content: `scheduled message: ${description}`,
      },
    ]);
  }
}

/**
 * Worker entry point that routes incoming requests to the appropriate handler
 */
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    if (!env.DEEPSEEK_API_KEY) {
      console.error(
        "DEEPSEEK_API_KEY is not set, don't forget to set it locally in .dev.vars, and use `wrangler secret bulk .dev.vars` to upload it to production"
      );
      return new Response("DEEPSEEK_API_KEY is not set", { status: 500 });
    }
    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  },
} satisfies ExportedHandler<Env>;
