import { useCallback, useEffect, useRef, useState } from "react";
import type { ExecutionMessage, ModelInfo, NodesByCategory } from "../types";

interface UseWebSocketResult {
  connected: boolean;
  models: Record<string, ModelInfo>;
  nodeDefinitions: NodesByCategory;
  executionMessages: ExecutionMessage[];
  isExecuting: boolean;
  sendMessage: (message: Record<string, unknown>) => void;
  executeWorkflow: (nodes: unknown[], edges: unknown[]) => void;
  cancelExecution: () => void;
  clearExecutionMessages: () => void;
}

export function useWebSocket(url: string): UseWebSocketResult {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [nodeDefinitions, setNodeDefinitions] = useState<NodesByCategory>({});
  const [executionMessages, setExecutionMessages] = useState<
    ExecutionMessage[]
  >([]);
  const [isExecuting, setIsExecuting] = useState(false);

  const handleMessage = useCallback((message: ExecutionMessage) => {
    switch (message.type) {
      case "models":
        setModels(message.data as Record<string, ModelInfo>);
        break;
      case "nodes":
        setNodeDefinitions(message.data as NodesByCategory);
        break;
      case "execution_start":
        setIsExecuting(true);
        setExecutionMessages((prev) => [...prev, message]);
        break;
      case "execution_complete":
      case "execution_error":
      case "execution_cancelled":
        setIsExecuting(false);
        setExecutionMessages((prev) => [...prev, message]);
        break;
      case "node_start":
      case "node_complete":
      case "node_error":
        setExecutionMessages((prev) => [...prev, message]);
        break;
    }
  }, []);

  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        setConnected(true);
        ws.current?.send(JSON.stringify({ type: "get_models" }));
        ws.current?.send(JSON.stringify({ type: "get_nodes" }));
      };

      ws.current.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };

      ws.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
      };
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, [url, handleMessage]);

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    }
  }, []);

  const executeWorkflow = useCallback(
    (nodes: unknown[], edges: unknown[]) => {
      setExecutionMessages([]);
      setIsExecuting(true);
      sendMessage({ type: "execute", nodes, edges });
    },
    [sendMessage],
  );

  const cancelExecution = useCallback(() => {
    sendMessage({ type: "cancel_execution" });
  }, [sendMessage]);

  const clearExecutionMessages = useCallback(() => {
    setExecutionMessages([]);
  }, []);

  return {
    connected,
    models,
    nodeDefinitions,
    executionMessages,
    isExecuting,
    sendMessage,
    executeWorkflow,
    cancelExecution,
    clearExecutionMessages,
  };
}
