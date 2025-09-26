"use client";

import React, { useEffect, useMemo } from "react";
import { User, Bot } from "lucide-react";
import { SubAgentIndicator } from "../SubAgentIndicator/SubAgentIndicator";
import { ToolCallBox } from "../ToolCallBox/ToolCallBox";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import type { SubAgent, ToolCall } from "../../types/types";
import styles from "./ChatMessage.module.scss";
import { Message } from "@langchain/langgraph-sdk";
import { extractStringFromMessageContent } from "../../utils/utils";

interface ChatMessageProps {
  message: Message;
  toolCalls: ToolCall[];
  showAvatar: boolean;
  onSelectSubAgent: (subAgent: SubAgent) => void;
  selectedSubAgent: SubAgent | null;
  /** NEW: dùng để dừng đồng hồ trong ToolCallBox khi stream kết thúc */
  isStreaming?: boolean;
}

export const ChatMessage = React.memo<ChatMessageProps>(
  ({
    message,
    toolCalls,
    showAvatar,
    onSelectSubAgent,
    selectedSubAgent,
    isStreaming, // <- NEW
  }) => {
    const isUser = message.type === "human";
    const messageContent = extractStringFromMessageContent(message);
    const hasContent = !!messageContent && messageContent.trim() !== "";
    const hasToolCalls = toolCalls.length > 0;

    // Gom các "task" subagents (nếu có) để hiển thị badges
    const subAgents = useMemo<SubAgent[]>(() => {
      return toolCalls
        .filter((tc) => tc.name === "task" && tc.args?.subagent_type)
        .map((tc) => ({
          id: tc.id,
          name: tc.name,
          subAgentName: tc.args["subagent_type"],
          input: tc.args["description"],
          output: tc.result,
          status: tc.status,
        }));
    }, [toolCalls]);

    // Duy trì lựa chọn subAgent nếu đang chọn
    const subAgentsString = useMemo(() => JSON.stringify(subAgents), [subAgents]);
    useEffect(() => {
      const found = subAgents.find((s) => s.id === selectedSubAgent?.id);
      if (found) onSelectSubAgent(found);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedSubAgent, subAgentsString]);

    return (
      <div className={`${styles.message} ${isUser ? styles.user : styles.assistant}`}>
        <div className={`${styles.avatar} ${!showAvatar ? styles.avatarHidden : ""}`}>
          {showAvatar &&
            (isUser ? (
              <User className={styles.avatarIcon} />
            ) : (
              <Bot className={styles.avatarIcon} />
            ))}
        </div>

        <div className={styles.content}>
          {hasContent && (
            <div className={styles.bubble}>
              {isUser ? (
                <p className={styles.text}>{messageContent}</p>
              ) : (
                <MarkdownContent content={messageContent} />
              )}
            </div>
          )}

          {hasToolCalls && (
            <div className={styles.toolCalls}>
              {toolCalls.map((tc) => {
                if (tc.name === "task") return null; // không render task ở khối ToolCall
                return (
                  <ToolCallBox
                    key={tc.id}
                    toolCall={tc}
                    /** NEW: truyền cờ streaming xuống box để chốt timer đúng lúc */
                    isStreaming={isStreaming}
                  />
                );
              })}
            </div>
          )}

          {!isUser && subAgents.length > 0 && (
            <div className={styles.subAgents}>
              {subAgents.map((sa) => (
                <SubAgentIndicator
                  key={sa.id}
                  subAgent={sa}
                  onClick={() => onSelectSubAgent(sa)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }
);

ChatMessage.displayName = "ChatMessage";
