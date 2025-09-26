// ChatInterface.tsx
"use client";

import React, { useState, useRef, useCallback, useMemo, useEffect, FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Bot, LoaderCircle, SquarePen, History } from "lucide-react";
import { ChatMessage } from "../ChatMessage/ChatMessage";
import { ThreadHistorySidebar } from "../ThreadHistorySidebar/ThreadHistorySidebar";
import type { SubAgent, TodoItem, ToolCall } from "../../types/types";
import { useChat } from "../../hooks/useChat";
import styles from "./ChatInterface.module.scss";
import { Message } from "@langchain/langgraph-sdk";
import { extractStringFromMessageContent } from "../../utils/utils";

interface ChatInterfaceProps {
  threadId: string | null;
  selectedSubAgent: SubAgent | null;
  setThreadId: (value: string | ((old: string | null) => string | null) | null) => void;
  onSelectSubAgent: (subAgent: SubAgent) => void;
  onTodosUpdate: (todos: TodoItem[]) => void;
  onFilesUpdate: (files: Record<string, string>) => void;
  onNewThread: () => void;
  isLoadingThreadState: boolean;
}

export const ChatInterface = React.memo<ChatInterfaceProps>(function ChatInterface({
  threadId,
  selectedSubAgent,
  setThreadId,
  onSelectSubAgent,
  onTodosUpdate,
  onFilesUpdate,
  onNewThread,
  isLoadingThreadState,
}) {
  const [input, setInput] = useState("");
  const [isThreadHistoryOpen, setIsThreadHistoryOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, isLoading, sendMessage, stopStream } = useChat(
    threadId, setThreadId, onTodosUpdate, onFilesUpdate,
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = useCallback((e: FormEvent) => {
    e.preventDefault();
    const messageText = input.trim();
    if (!messageText || isLoading) return;
    sendMessage(messageText);
    setInput("");
  }, [input, isLoading, sendMessage]);

  const handleNewThread = useCallback(() => {
    if (isLoading) stopStream();
    setIsThreadHistoryOpen(false);
    onNewThread();
  }, [isLoading, stopStream, onNewThread]);

  const handleThreadSelect = useCallback((id: string) => {
    setThreadId(id);
    setIsThreadHistoryOpen(false);
  }, [setThreadId]);

  const toggleThreadHistory = useCallback(() => setIsThreadHistoryOpen((p) => !p), []);

  const hasMessages = messages.length > 0;
  const toolTimers = useRef<Record<string, number>>({}); // fallback timer nếu backend chưa kịp gửi elapsed_ms

  const processedMessages = useMemo(() => {
    const map = new Map<string, { message: Message; toolCalls: ToolCall[] }>();

    const ensureId = (v?: string) =>
      v || (globalThis.crypto?.randomUUID?.() ?? `tool-${Math.random().toString(36).slice(2)}`);

    messages.forEach((m: Message) => {
      // AI => tìm tool_use / tool_calls để tạo pending
      if (m.type === "ai") {
        const tcs: any[] = [];
        if (m.additional_kwargs?.tool_calls && Array.isArray(m.additional_kwargs.tool_calls)) {
          tcs.push(...m.additional_kwargs.tool_calls);
        } else if ((m as any).tool_calls && Array.isArray((m as any).tool_calls)) {
          tcs.push(...(m as any).tool_calls);
        } else if (Array.isArray(m.content)) {
          tcs.push(...m.content.filter((b: any) => b?.type === "tool_use"));
        }

        const now = Date.now();
        const toolCalls: ToolCall[] = tcs.map((tc: any) => {
          const id = ensureId(tc.id || tc.tool_call_id);
          const name = tc.function?.name || tc.name || tc.type || "unknown";
          const args = tc.function?.arguments || tc.args || tc.input || {};
          if (!toolTimers.current[id]) toolTimers.current[id] = now;
          return { id, name, args, status: "pending", startedAt: now };
        });

        map.set(m.id!, { message: m, toolCalls });
        return;
      }

      // TOOL => match theo tool_call_id, cập nhật completed + elapsed_ms
      if (m.type === "tool") {
        const tcid = (m as any).tool_call_id;
        if (!tcid) return;

        for (const [, data] of map.entries()) {
          const idx = data.toolCalls.findIndex((t) => t.id === tcid);
          if (idx === -1) continue;

          const start = toolTimers.current[tcid];
          const elapsedFromTimer = start ? Date.now() - start : undefined;
          const elapsedFromBackend = (m as any).additional_kwargs?.elapsed_ms;

          data.toolCalls[idx] = {
            ...data.toolCalls[idx],
            status: "completed",
            result: extractStringFromMessageContent(m),
            elapsedMs: typeof elapsedFromBackend === "number" ? elapsedFromBackend : elapsedFromTimer,
          };
          delete toolTimers.current[tcid];
          break;
        }
        return;
      }

      // HUMAN => giữ bubble user
      if (m.type === "human") {
        map.set(m.id!, { message: m, toolCalls: [] });
      }
    });

    const arr = Array.from(map.values());
    return arr.map((d, i) => {
      const prev = i > 0 ? arr[i - 1].message : null;
      return { ...d, showAvatar: d.message.type !== prev?.type };
    });
  }, [messages]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <Bot className={styles.logo} />
          <h1 className={styles.title}>Deep Agents</h1>
        </div>
        <div className={styles.headerRight}>
          <Button variant="ghost" size="icon" onClick={handleNewThread} disabled={!hasMessages} title="New thread">
            <SquarePen size={20} />
          </Button>
          <Button variant="ghost" size="icon" onClick={toggleThreadHistory} title="Thread history">
            <History size={20} />
          </Button>
        </div>
      </div>

      <div className={styles.content}>
        <ThreadHistorySidebar
          open={isThreadHistoryOpen}
          setOpen={setIsThreadHistoryOpen}
          currentThreadId={threadId}
          onThreadSelect={handleThreadSelect}
        />

        <div className={styles.messagesContainer}>
          {!hasMessages && !isLoading && !isLoadingThreadState && (
            <div className={styles.emptyState}>
              <Bot size={48} className={styles.emptyIcon} />
              <h2>Start a conversation or select a thread from history</h2>
            </div>
          )}

          {isLoadingThreadState && (
            <div className={styles.threadLoadingState}>
              <LoaderCircle className={styles.threadLoadingSpinner} />
            </div>
          )}

          <div className={styles.messagesList}>
            {processedMessages.map((d) => (
              <ChatMessage
                key={d.message.id}
                message={d.message}
                toolCalls={d.toolCalls}
                showAvatar={d.showAvatar}
                onSelectSubAgent={onSelectSubAgent}
                selectedSubAgent={selectedSubAgent}
                isStreaming={isLoading}
              />
            ))}

            {isLoading && (
              <div className={styles.loadingMessage}>
                <LoaderCircle className={styles.spinner} />
                <span>Working...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className={styles.inputForm}>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          className={styles.input}
        />
        {isLoading ? (
          <Button type="button" onClick={stopStream} className={styles.stopButton}>Stop</Button>
        ) : (
          <Button type="submit" disabled={!input.trim()} className={styles.sendButton}>
            <Send size={16} />
          </Button>
        )}
      </form>
    </div>
  );
});
ChatInterface.displayName = "ChatInterface";
