import { useCallback, useEffect, useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { Sidebar } from "./components/Sidebar";
import { Toolbar } from "./components/Toolbar";
import { Canvas } from "./components/Canvas";
import { PropertiesPanel } from "./components/PropertiesPanel";
import { ErrorToast } from "./components/ErrorToast";
import { TournamentView } from "./components/TournamentView";
import { useWebSocket } from "./hooks/useWebSocket";
import { useWorkflowStore } from "./store/workflowStore";
import type { ErrorInfo } from "./types";
import diamondTournamentYaml from "./templates/diamond-tournament.yml?raw";
import "./App.css";

type Tab = "workflow" | "results";

const WS_PROTOCOL = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${WS_PROTOCOL}//${window.location.host}/ws`;

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("workflow");

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
    if (Object.keys(nodeDefinitions).length === 0 || nodes.length > 0) return;
    (async () => {
      try {
        const yaml = await import("js-yaml");
        const data = yaml.load(diamondTournamentYaml) as {
          nodes: unknown[];
          edges: unknown[];
        };
        loadWorkflow(data, nodeDefinitions);
      } catch (err) {
        console.error(
          "Failed to load Diamond Tournament template, falling back",
          err,
        );
        loadStartupWorkflow(nodeDefinitions);
      }
    })();
  }, [nodeDefinitions, nodes.length, loadStartupWorkflow, loadWorkflow]);

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
      <div className="tab-bar">
        <button
          type="button"
          className={activeTab === "workflow" ? "active" : ""}
          onClick={() => setActiveTab("workflow")}
        >
          Workflow Editor
        </button>
        <button
          type="button"
          className={activeTab === "results" ? "active" : ""}
          onClick={() => setActiveTab("results")}
        >
          Tournament Results
        </button>
      </div>
      {activeTab === "workflow" ? (
        <>
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
        </>
      ) : (
        <TournamentView />
      )}
    </div>
  );
}

export default App;
