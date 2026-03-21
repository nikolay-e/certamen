import { useCallback, useEffect, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { Sidebar } from "./components/Sidebar";
import { Toolbar } from "./components/Toolbar";
import { Canvas } from "./components/Canvas";
import { PropertiesPanel } from "./components/PropertiesPanel";
import { ErrorToast } from "./components/ErrorToast";
import { useWebSocket } from "./hooks/useWebSocket";
import { useWorkflowStore } from "./store/workflowStore";
import type { ErrorInfo } from "./types";
import "./App.css";

const WS_PROTOCOL = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${WS_PROTOCOL}//${window.location.host}/ws`;

function App() {
  const {
    connected,
    models,
    nodeDefinitions,
    executionMessages,
    isExecuting,
    executeWorkflow,
    cancelExecution,
    clearExecutionMessages,
  } = useWebSocket(WS_URL);

  const currentError: ErrorInfo | null = useMemo(() => {
    for (let i = executionMessages.length - 1; i >= 0; i--) {
      const msg = executionMessages[i];
      if (msg.type === "execution_error" && msg.error) {
        return typeof msg.error === "string"
          ? { type: "Error", message: msg.error }
          : msg.error;
      }
      if (msg.type === "execution_start") {
        return null;
      }
    }
    return null;
  }, [executionMessages]);

  const {
    nodes,
    getWorkflowData,
    clearWorkflow,
    updateNodeData,
    updateNodeProperty,
    loadStartupWorkflow,
    loadWorkflow,
    setDefaultModelIfEmpty,
  } = useWorkflowStore();

  const selectedNodeData = useWorkflowStore(
    useShallow((state) => {
      if (!state.selectedNode) return null;
      const node = state.nodes.find((n) => n.id === state.selectedNode);
      if (!node) return null;
      return { id: node.id, data: node.data };
    }),
  );

  useEffect(() => {
    if (Object.keys(nodeDefinitions).length > 0 && nodes.length === 0) {
      loadStartupWorkflow(nodeDefinitions);
    }
  }, [nodeDefinitions, nodes.length, loadStartupWorkflow]);

  useEffect(() => {
    if (Object.keys(models).length > 0) {
      setDefaultModelIfEmpty(models);
    }
  }, [models, setDefaultModelIfEmpty]);

  const handleExecute = useCallback(() => {
    nodes.forEach((node) => {
      updateNodeData(node.id, {
        executing: false,
        completed: false,
        error: undefined,
        result: undefined,
      });
    });
    clearExecutionMessages();
    const { nodes: workflowNodes, edges } = getWorkflowData();
    executeWorkflow(workflowNodes, edges);
  }, [
    nodes,
    updateNodeData,
    clearExecutionMessages,
    getWorkflowData,
    executeWorkflow,
  ]);

  const handleClear = useCallback(() => {
    clearWorkflow();
    clearExecutionMessages();
  }, [clearWorkflow, clearExecutionMessages]);

  const handlePropertyChange = useCallback(
    (nodeId: string, key: string, value: unknown) => {
      updateNodeProperty(nodeId, key, value);
    },
    [updateNodeProperty],
  );

  const handleLoadWorkflow = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".yaml,.yml";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      try {
        const text = await file.text();
        const yaml = await import("js-yaml");
        const workflowData = yaml.load(text) as {
          nodes: unknown[];
          edges: unknown[];
        };
        loadWorkflow(workflowData, nodeDefinitions);
      } catch (error) {
        console.error("Failed to load workflow:", error);
        alert(
          `Failed to load workflow: ${
            error instanceof Error ? error.message : "Unknown error"
          }`,
        );
      }
    };
    input.click();
  }, [loadWorkflow, nodeDefinitions]);

  const handleSaveWorkflow = useCallback(async () => {
    const { nodes: workflowNodes, edges } = getWorkflowData();
    const workflow = {
      name: "Workflow",
      description: "",
      nodes: workflowNodes,
      edges: edges,
    };

    const yaml = await import("js-yaml");
    const yamlText = yaml.dump(workflow, { indent: 2, lineWidth: -1 });
    const blob = new Blob([yamlText], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workflow.yml";
    a.click();
    URL.revokeObjectURL(url);
  }, [getWorkflowData]);

  return (
    <div className="app">
      <Toolbar
        onExecute={handleExecute}
        onCancel={cancelExecution}
        onClear={handleClear}
        onLoadWorkflow={handleLoadWorkflow}
        onSaveWorkflow={handleSaveWorkflow}
        executing={isExecuting}
        hasNodes={nodes.length > 0}
      />
      <div className="main-content">
        <Sidebar nodeDefinitions={nodeDefinitions} connected={connected} />
        <Canvas executionMessages={executionMessages} />
        <PropertiesPanel
          node={selectedNodeData}
          models={models}
          onPropertyChange={handlePropertyChange}
        />
      </div>
      <ErrorToast
        error={currentError}
        onClose={() => clearExecutionMessages()}
      />
    </div>
  );
}

export default App;
