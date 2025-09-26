// ToolCallBox.tsx
"use client";

import React, { useMemo, useState, useEffect } from "react";
import { ChevronDown, ChevronRight, CheckCircle, AlertCircle, Loader } from "lucide-react";
import { Button } from "@/components/ui/button";
import styles from "./ToolCallBox.module.scss";
import type { ToolCall } from "../../types/types";

function formatMs(ms?: number) {
  if (typeof ms !== "number" || !isFinite(ms) || ms < 0) return "0.0 s";
  return `${(ms / 1000).toFixed(1)} s`;
}

interface ToolCallBoxProps {
  toolCall: ToolCall;
  isStreaming?: boolean;
}


export const ToolCallBox = React.memo<ToolCallBoxProps>(({ toolCall, isStreaming }) => {
  const [open, setOpen] = useState(false);
  const toggle = () => setOpen(v => !v);

  // reset local timers mỗi khi toolCall đổi (tránh giữ state cũ)
  const [tick, setTick] = useState(0);
  const [freezeAt, setFreezeAt] = useState<number | null>(null);
  useEffect(() => {
    setTick(0);
    setFreezeAt(null);
  }, [toolCall.id]);

  const icon = useMemo(() => {
    switch (toolCall.status) {
      case "completed":
        return <CheckCircle className={styles.statusCompleted} aria-label="Completed" />;
      case "error":
        return <AlertCircle className={styles.statusError} aria-label="Error" />;
      case "pending":
      default:
        return <Loader className={styles.statusRunning} aria-label="Running" />;
    }
  }, [toolCall.status]);

  // Cập nhật mỗi 200ms khi còn pending & đang stream
  useEffect(() => {
    if (toolCall.status === "pending" && toolCall.startedAt && isStreaming) {
      const id = setInterval(() => setTick(Date.now()), 200);
      return () => clearInterval(id);
    }
  }, [toolCall.status, toolCall.startedAt, isStreaming]);

  // Khi stream dừng mà tool vẫn pending => freeze đồng hồ
  useEffect(() => {
    if (!isStreaming && toolCall.status === "pending" && toolCall.startedAt && freezeAt == null) {
      setFreezeAt(Date.now());
    }
  }, [isStreaming, toolCall.status, toolCall.startedAt, freezeAt]);

  // Hiển thị thời gian:
  // - ưu tiên elapsedMs do backend gửi (chính xác như LangSmith)
  // - nếu pending & đang streaming -> đếm theo startedAt
  const shownElapsed = useMemo(() => {
    if (typeof toolCall.elapsedMs === "number") return toolCall.elapsedMs;

    if (toolCall.doneAt && toolCall.startedAt) {
      return Math.max(0, toolCall.doneAt - toolCall.startedAt);
    }

    if (toolCall.status === "pending" && toolCall.startedAt) {
      const endRef = isStreaming ? Date.now() : (freezeAt ?? Date.now());
      return Math.max(0, endRef - toolCall.startedAt);
    }

    return 0;
  }, [
    toolCall.elapsedMs,
    toolCall.doneAt,
    toolCall.startedAt,
    toolCall.status,
    isStreaming,
    freezeAt,
    tick,
  ]);

  const hasArgs = toolCall.args && Object.keys(toolCall.args || {}).length > 0;
  const hasResult = !!toolCall.result;

  return (
    <div className={styles.container}>
      <Button
        variant="ghost"
        size="sm"
        onClick={toggle}
        className={styles.header}
        disabled={!hasArgs && !hasResult}
      >
        <div className={styles.headerLeft}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {icon}
          <span className={styles.toolName}>{toolCall.name}</span>
          <span className={styles.toolTime}>
            {toolCall.status} • {formatMs(shownElapsed)}
          </span>
        </div>
      </Button>

      {open && (hasArgs || hasResult) && (
        <div className={styles.content}>
          {hasArgs && (
            <div className={styles.section}>
              <h4 className={styles.sectionTitle}>Arguments</h4>
              <pre className={styles.codeBlock}>{JSON.stringify(toolCall.args, null, 2)}</pre>
            </div>
          )}
          {hasResult && (
            <div className={styles.section}>
              <h4 className={styles.sectionTitle}>Result</h4>
              <pre className={styles.codeBlock}>
                {typeof toolCall.result === "string" ? toolCall.result : JSON.stringify(toolCall.result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

ToolCallBox.displayName = "ToolCallBox";
