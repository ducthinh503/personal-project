// frontend/src/app/components/hooks/useChat.ts
import { useCallback, useMemo } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";

import { getDeployment } from "@/lib/environment/deployments";
import type { TodoItem } from "@/app/types/types";
import { createClient } from "@/lib/client";
import { useAuthContext } from "@/providers/Auth";

type GraphState = {
  // state bạn muốn nhận lại từ server
  messages?: Message[];
  todos?: TodoItem[];
  files?: Record<string, string>;
  // THÊM: để submit { input: string } không bị lỗi kiểu
  input?: string;
};

export function useChat(
  threadId: string | null,
  setThreadId: (
    value: string | ((old: string | null) => string | null) | null,
  ) => void,
  onTodosUpdate: (todos: TodoItem[]) => void,
  onFilesUpdate: (files: Record<string, string>) => void,
) {
  const deployment = useMemo(() => getDeployment(), []);
  const { session } = useAuthContext();
  const accessToken = session?.accessToken ?? "";

  const assistantId = useMemo(() => {
    if (!deployment?.agentId) {
      throw new Error("No agent ID configured in environment");
    }
    return deployment.agentId;
  }, [deployment]);

  const stream = useStream<GraphState>({
    assistantId,
    client: createClient(accessToken),
    reconnectOnMount: true,
    threadId: threadId ?? undefined,
    onThreadId: setThreadId, // server tạo thread mới => cập nhật URL
    defaultHeaders: { "x-auth-scheme": "langsmith" },

    // nhận partial state từ server (nếu có) để cập nhật sidebar
    onUpdateEvent: (updates) => {
      Object.values(updates).forEach((u) => {
        if (u?.todos) onTodosUpdate(u.todos);
        if (u?.files) onFilesUpdate(u.files);
      });
    },
  });

  const sendMessage = useCallback(
    (text: string) => {
      const human: Message = { id: uuidv4(), type: "human", content: text };

      // ✅ backend (main.py) đang đọc state["input"] (string)
      stream.submit(
        { input: text } as any,
        {
          // show ngay bubble của user
          optimisticValues(prev) {
            const prevMsgs = (prev.messages ?? []) as Message[];
            return { ...prev, messages: [...prevMsgs, human] };
          },
          config: { recursion_limit: 100 },
        },
      );
    },
    [stream],
  );

  const stopStream = useCallback(() => stream.stop(), [stream]);

  return {
    messages: stream.messages,
    isLoading: stream.isLoading,
    sendMessage,
    stopStream,
  };
}
