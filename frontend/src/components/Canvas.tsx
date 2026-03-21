import { useCallback, useEffect, useRef, useMemo } from "react";
import type { DragEvent } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import type { Connection, Edge, EdgeTypes } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useWorkflowStore } from "../store/workflowStore";
import { BaseNode } from "../nodes/BaseNode";
import { ColoredSmartEdge } from "./ColoredSmartEdge";
import type { NodeDefinition, ExecutionMessage, NodeData } from "../types";
import { isPortCompatible } from "../utils";
import { NODE_TYPES } from "../constants/nodeTypes";

const nodeTypes = {
  [NODE_TYPES.WORKFLOW]: BaseNode,
};

const edgeTypes: EdgeTypes = {
  colored: ColoredSmartEdge,
};

interface CanvasProps {
  executionMessages: ExecutionMessage[];
}

function CanvasInner({ executionMessages }: CanvasProps) {
  const { screenToFlowPosition } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  const {
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    updateNodeData,
    updateNodeProperty,
    setSelectedNode,
    clearNodesNeedingUpdate,
  } = useWorkflowStore();

  const {
    nodes,
    edges: rawEdges,
    nodeIdsNeedingUpdate,
  } = useWorkflowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      nodeIdsNeedingUpdate: state.nodeIdsNeedingUpdate,
    })),
  );

  const edges = useMemo(() => {
    return rawEdges.map((edge) => ({
      ...edge,
      type: "colored",
    }));
  }, [rawEdges]);

  const validationTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  const validateWorkflow = useCallback(async () => {
    try {
      const response = await fetch("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodes, edges }),
      });

      if (response.ok) {
        const data = (await response.json()) as {
          valid: boolean;
          warnings?: Array<{ node_id: string; message: string }>;
        };

        if (data.warnings && data.warnings.length > 0) {
          for (const warning of data.warnings) {
            updateNodeData(warning.node_id, {
              hasWarning: true,
              warning: warning.message,
            });
          }
        } else {
          for (const node of nodes) {
            updateNodeData(node.id, {
              hasWarning: false,
              warning: undefined,
            });
          }
        }
      }
    } catch (error) {
      console.error("Validation error:", error);
    }
  }, [nodes, edges, updateNodeData]);

  useEffect(() => {
    if (validationTimeoutRef.current) {
      clearTimeout(validationTimeoutRef.current);
    }

    validationTimeoutRef.current = setTimeout(() => {
      validateWorkflow();
    }, 500);

    return () => {
      if (validationTimeoutRef.current) {
        clearTimeout(validationTimeoutRef.current);
      }
    };
  }, [nodes, edges, validateWorkflow]);

  const isValidConnection = useCallback(
    (connection: Edge | Connection) => {
      if (!connection.source || !connection.target) return false;
      if (!connection.sourceHandle || !connection.targetHandle) return false;

      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);

      if (!sourceNode || !targetNode) return false;

      const sourcePort = sourceNode.data.outputs.find(
        (p) => p.name === connection.sourceHandle!,
      );
      const targetPort = targetNode.data.inputs.find(
        (p) => p.name === connection.targetHandle!,
      );

      if (!sourcePort || !targetPort) return false;

      return isPortCompatible(sourcePort, targetPort);
    },
    [nodes],
  );

  useEffect(() => {
    for (const msg of executionMessages) {
      if (msg.node_id) {
        switch (msg.type) {
          case "node_start":
            updateNodeData(msg.node_id, {
              executing: true,
              completed: false,
              error: undefined,
            });
            break;
          case "node_complete": {
            const outputs = msg.outputs as Record<string, unknown> | undefined;

            // Update pages for nodes that output _pages (e.g., TextNode)
            // This avoids depending on nodes array which may be stale
            if (outputs?._pages) {
              const pages = outputs._pages as string[][] | undefined;
              const totalPages = outputs._total_pages as number | undefined;

              if (pages && Array.isArray(pages) && pages.length > 0) {
                updateNodeProperty(msg.node_id, "pages", pages);
                // Ensure current_page is within bounds of actual pages array
                const pageIndex = Math.min(
                  Math.max(0, (totalPages ?? pages.length) - 1),
                  pages.length - 1,
                );
                updateNodeProperty(msg.node_id, "current_page", pageIndex);
              }
            }

            updateNodeData(msg.node_id, {
              executing: false,
              completed: true,
              result: outputs,
            });
            break;
          }
          case "node_error":
            updateNodeData(msg.node_id, {
              executing: false,
              completed: false,
              error: msg.error,
            });
            break;
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [executionMessages]);

  // Update React Flow internals when dynamic handles are added
  useEffect(() => {
    if (nodeIdsNeedingUpdate.length > 0) {
      for (const nodeId of nodeIdsNeedingUpdate) {
        updateNodeInternals(nodeId);
      }
      clearNodesNeedingUpdate();
    }
  }, [nodeIdsNeedingUpdate, updateNodeInternals, clearNodesNeedingUpdate]);

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      const data = event.dataTransfer.getData("application/reactflow");
      if (!data) return;

      const nodeDef: NodeDefinition = JSON.parse(data);
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode = {
        id: `${nodeDef.node_type}-${Date.now()}`,
        type: NODE_TYPES.WORKFLOW,
        position,
        data: {
          label: nodeDef.display_name,
          nodeType: nodeDef.node_type,
          category: nodeDef.category,
          inputs: [...nodeDef.inputs],
          outputs: nodeDef.outputs,
          properties: Object.fromEntries(
            Object.entries(nodeDef.properties).map(([key, prop]) => [
              key,
              prop.default,
            ]),
          ),
          propertyDefs: nodeDef.properties,
          dynamicInputs: nodeDef.dynamic_inputs,
        } as NodeData,
      };

      addNode(newNode);
    },
    [screenToFlowPosition, addNode],
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNode(node.id);
    },
    [setSelectedNode],
  );

  return (
    <div className="canvas-container">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={{ type: "colored" }}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        minZoom={0.1}
        maxZoom={4}
      >
        <Background gap={15} size={1} />
        <Controls />
        <MiniMap
          pannable
          zoomable
          nodeColor={(node) => {
            const nodeData = node.data as NodeData | undefined;
            const category = nodeData?.category;
            const colors: Record<string, string> = {
              input: "#3b82f6",
              generation: "#10b981",
              evaluation: "#f59e0b",
              flow: "#8b5cf6",
              output: "#ec4899",
            };
            return category ? colors[category] || "#6b7280" : "#6b7280";
          }}
        />
      </ReactFlow>
    </div>
  );
}

export function Canvas(props: CanvasProps) {
  return (
    <ReactFlowProvider>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  );
}
