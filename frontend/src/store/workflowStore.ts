import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { applyNodeChanges, applyEdgeChanges, addEdge } from "@xyflow/react";
import type {
  Node,
  Edge,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  Connection,
} from "@xyflow/react";
import type { NodeData } from "../types";
import { NODE_TYPES, validateNodeType } from "../constants/nodeTypes";

type WorkflowNode = Node<NodeData>;

interface WorkflowState {
  nodes: WorkflowNode[];
  edges: Edge[];
  selectedNode: string | null;
  nodeIdsNeedingUpdate: string[];
}

interface WorkflowActions {
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  addNode: (node: WorkflowNode) => void;
  deleteNode: (nodeId: string) => void;
  updateNodeData: (nodeId: string, data: Partial<NodeData>) => void;
  updateNodeProperty: (nodeId: string, key: string, value: unknown) => void;
  setSelectedNode: (nodeId: string | null) => void;
  clearNodesNeedingUpdate: () => void;
  getWorkflowData: () => { nodes: unknown[]; edges: unknown[] };
  clearWorkflow: () => void;
  loadStartupWorkflow: (nodeDefinitions: Record<string, unknown[]>) => void;
  loadWorkflow: (
    workflowJson: { nodes: unknown[]; edges: unknown[] },
    nodeDefinitions: Record<string, unknown[]>,
  ) => void;
  setDefaultModelIfEmpty: (models: Record<string, unknown>) => void;
}

type WorkflowStore = WorkflowState & WorkflowActions;

const initialNodes: WorkflowNode[] = [];

const initialEdges: Edge[] = [];

export const useWorkflowStore = create<WorkflowStore>()(
  subscribeWithSelector((set, get) => ({
  nodes: initialNodes,
  edges: initialEdges,
  selectedNode: null,
  nodeIdsNeedingUpdate: [],

  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) as WorkflowNode[] });
  },

  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect: (connection: Connection) => {
    const { nodes, edges } = get();
    const newEdges = addEdge(connection, edges);

    // Check if we need to add a dynamic port
    const targetNode = nodes.find((n) => n.id === connection.target);
    if (targetNode?.data.dynamicInputs && connection.targetHandle) {
      const { prefix, port_type } = targetNode.data.dynamicInputs;

      // Check if the connected port matches the dynamic prefix
      if (connection.targetHandle.startsWith(`${prefix}_`)) {
        // Count existing ports with this prefix
        const existingPorts = targetNode.data.inputs.filter((p) =>
          p.name.startsWith(`${prefix}_`),
        );

        // Check if all existing dynamic ports are connected
        const connectedPorts = new Set(
          newEdges
            .filter((e) => e.target === targetNode.id)
            .map((e) => e.targetHandle),
        );

        const allConnected = existingPorts.every((p) =>
          connectedPorts.has(p.name),
        );

        if (allConnected) {
          // Add a new dynamic port
          const newPortNumber = existingPorts.length + 1;
          const newPort = {
            name: `${prefix}_${newPortNumber}`,
            port_type: port_type,
            required: false,
            description: `${prefix.charAt(0).toUpperCase() + prefix.slice(1)} ${newPortNumber}`,
          };

          const updatedNodes = nodes.map((n) =>
            n.id === targetNode.id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    inputs: [...n.data.inputs, newPort],
                  },
                }
              : n,
          );

          // Mark node as needing React Flow internal update
          set({
            nodes: updatedNodes,
            edges: newEdges,
            nodeIdsNeedingUpdate: [
              ...get().nodeIdsNeedingUpdate,
              targetNode.id,
            ],
          });
          return;
        }
      }
    }

    set({ edges: newEdges });
  },

  addNode: (node) => {
    set({ nodes: [...get().nodes, node] });
  },

  deleteNode: (nodeId) => {
    set({
      nodes: get().nodes.filter((node) => node.id !== nodeId),
      edges: get().edges.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId,
      ),
      selectedNode: get().selectedNode === nodeId ? null : get().selectedNode,
    });
  },

  updateNodeData: (nodeId, data) => {
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node,
      ),
    });
  },

  updateNodeProperty: (nodeId, key, value) => {
    set({
      nodes: get().nodes.map((node) => {
        if (node.id !== nodeId) return node;

        const newProperties = { ...node.data.properties, [key]: value };

        // For text nodes: clear execution pages when texts are edited
        // This ensures canvas shows user input, not stale execution results
        if (node.data.nodeType === "simple/text" && key === "texts") {
          newProperties.pages = [];
          newProperties.current_page = 0;
        }

        return {
          ...node,
          data: {
            ...node.data,
            properties: newProperties,
          },
        };
      }),
    });
  },

  setSelectedNode: (nodeId) => {
    set({ selectedNode: nodeId });
  },

  clearNodesNeedingUpdate: () => {
    set({ nodeIdsNeedingUpdate: [] });
  },

  getWorkflowData: () => {
    const { nodes, edges } = get();
    return {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.nodeType,
        position: n.position,
        properties: n.data.properties, // properties на верхнем уровне, не в data!
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
      })),
    };
  },

  clearWorkflow: () => {
    set({ nodes: [], edges: [], selectedNode: null });
  },

  loadStartupWorkflow: (nodeDefinitions) => {
    const simpleNodes = (nodeDefinitions.Simple ||
      nodeDefinitions.simple ||
      nodeDefinitions.LLM ||
      nodeDefinitions.llm ||
      []) as Record<string, unknown>[];
    const textNodeDef = simpleNodes.find((n) => n.node_type === "simple/text");
    const llmNodeDef = simpleNodes.find((n) => n.node_type === "simple/llm");

    if (!textNodeDef || !llmNodeDef) {
      console.warn("Cannot load startup workflow: node definitions not found");
      return;
    }

    const createNode = (
      def: Record<string, unknown>,
      id: string,
      position: { x: number; y: number },
      customProps?: Record<string, unknown>,
    ) => {
      const nodeType = NODE_TYPES.WORKFLOW;
      validateNodeType(nodeType);

      return {
        id,
        type: nodeType,
        position,
        data: {
          label: def.display_name,
          nodeType: def.node_type,
          category: def.category,
          inputs: def.inputs,
          outputs: def.outputs,
          properties: {
            ...Object.fromEntries(
              Object.entries(
                def.properties as Record<string, Record<string, unknown>>,
              ).map(([key, prop]) => [key, prop.default]),
            ),
            ...customProps,
          },
          propertyDefs: def.properties,
        } as NodeData,
      };
    };

    const nodes: WorkflowNode[] = [
      createNode(
        textNodeDef as Record<string, unknown>,
        "text-input",
        { x: 100, y: 150 },
        {
          texts: ["Say hello world in a creative and friendly way!"],
          separator: "\n",
          hidden: false,
        },
      ),
      createNode(
        llmNodeDef as Record<string, unknown>,
        "llm-main",
        { x: 450, y: 150 },
        {
          provider: "ollama",
        },
      ),
      createNode(
        textNodeDef as Record<string, unknown>,
        "text-output",
        { x: 800, y: 150 },
        {
          texts: [],
          separator: "\n",
          hidden: false,
        },
      ),
    ];

    const edges: Edge[] = [
      {
        id: "e-input-llm",
        source: "text-input",
        target: "llm-main",
        sourceHandle: "output_text",
        targetHandle: "prompt",
      },
      {
        id: "e-llm-output",
        source: "llm-main",
        target: "text-output",
        sourceHandle: "response",
        targetHandle: "input_text",
      },
    ];

    set({ nodes, edges });
  },

  loadWorkflow: (workflowJson, nodeDefinitions) => {
    const findNodeDefinition = (
      nodeType: string,
    ): Record<string, unknown> | null => {
      for (const category of Object.values(nodeDefinitions)) {
        const def = (category as Record<string, unknown>[]).find(
          (n) => n.node_type === nodeType,
        );
        if (def) return def;
      }
      return null;
    };

    const enrichNode = (
      jsonNode: Record<string, unknown>,
    ): WorkflowNode | null => {
      const backendNodeType =
        jsonNode.type || (jsonNode.data as Record<string, unknown>)?.nodeType;
      if (!backendNodeType) {
        console.warn(`Node ${jsonNode.id} missing type/nodeType, skipping`);
        return null;
      }

      const nodeDef = findNodeDefinition(backendNodeType as string);
      if (!nodeDef) {
        console.warn(
          `Node definition not found for type ${backendNodeType}, skipping node ${jsonNode.id}`,
        );
        return null;
      }

      const properties = ((jsonNode.properties as Record<string, unknown>) ||
        ((jsonNode.data as Record<string, unknown>)?.properties as Record<
          string,
          unknown
        >) ||
        {}) as Record<string, unknown>;
      const enrichedProperties = {
        ...Object.fromEntries(
          Object.entries(
            nodeDef.properties as Record<string, Record<string, unknown>>,
          ).map(([key, prop]) => [
            key,
            properties[key] !== undefined ? properties[key] : prop.default,
          ]),
        ),
        ...properties,
      };

      const xyflowNodeType = NODE_TYPES.WORKFLOW;
      validateNodeType(xyflowNodeType);

      return {
        id: jsonNode.id as string,
        type: xyflowNodeType,
        position: (jsonNode.position as { x: number; y: number }) || {
          x: 0,
          y: 0,
        },
        data: {
          label:
            ((jsonNode.data as Record<string, unknown>)?.label as string) ||
            (nodeDef.display_name as string),
          nodeType: nodeDef.node_type,
          category: nodeDef.category,
          inputs: [...(nodeDef.inputs as unknown[])],
          outputs: nodeDef.outputs,
          properties: enrichedProperties,
          propertyDefs: nodeDef.properties,
          dynamicInputs: nodeDef.dynamic_inputs,
        } as NodeData,
      };
    };

    const enrichedNodes = (workflowJson.nodes as Record<string, unknown>[])
      .map((node) => enrichNode(node))
      .filter((node): node is WorkflowNode => node !== null);

    const enrichedEdges = (
      (workflowJson.edges as Record<string, unknown>[]) || []
    ).map((edge) => ({
      id: (edge.id as string) || `${edge.source}-${edge.target}`,
      source: edge.source as string,
      target: edge.target as string,
      sourceHandle: edge.sourceHandle as string,
      targetHandle: edge.targetHandle as string,
    }));

    // Restore dynamic ports based on edges
    const nodeIdsNeedingUpdate: string[] = [];
    const nodesWithDynamicPorts = enrichedNodes.map((node) => {
      if (!node.data.dynamicInputs) return node;

      const { prefix, port_type, min_count } = node.data.dynamicInputs;

      // Find all edges targeting this node (both old and new naming)
      const allTargetEdges = enrichedEdges.filter((e) => e.target === node.id);

      console.log(`[Dynamic Ports] Node ${node.id}:`, {
        prefix,
        allTargetEdges: allTargetEdges.map((e) => e.targetHandle),
      });

      // Find edges with new naming (prefix_N)
      const newStyleHandles = allTargetEdges
        .filter((e) => e.targetHandle?.startsWith(`${prefix}_`))
        .map((e) => e.targetHandle);

      // Find edges with old naming (prefixN without underscore)
      const oldStyleHandles = allTargetEdges
        .filter((e) => e.targetHandle?.match(new RegExp(`^${prefix}\\d+$`)))
        .map((e) => e.targetHandle);

      // Determine which naming style to use (old vs new)
      const useOldStyle =
        oldStyleHandles.length > 0 && newStyleHandles.length === 0;

      console.log(`[Dynamic Ports] Naming analysis:`, {
        oldStyleCount: oldStyleHandles.length,
        newStyleCount: newStyleHandles.length,
        useOldStyle,
      });

      // Find the highest port number from edges (both styles)
      let maxPortNumber = 0;

      // Check new style (var_1, var_2)
      for (const handle of newStyleHandles) {
        const match = handle?.match(new RegExp(`^${prefix}_(\\d+)$`));
        if (match) {
          const num = parseInt(match[1], 10);
          if (num > maxPortNumber) maxPortNumber = num;
        }
      }

      // Check old style (var1, var2)
      for (const handle of oldStyleHandles) {
        const match = handle?.match(new RegExp(`^${prefix}(\\d+)$`));
        if (match) {
          const num = parseInt(match[1], 10);
          if (num > maxPortNumber) maxPortNumber = num;
        }
      }

      // Get current port count (check both naming styles)
      const currentPortsNew = node.data.inputs.filter((p) =>
        p.name.startsWith(`${prefix}_`),
      );
      const currentPortsOld = node.data.inputs.filter((p) =>
        p.name.match(new RegExp(`^${prefix}\\d+$`)),
      );
      const currentMax = Math.max(
        currentPortsNew.length,
        currentPortsOld.length,
      );

      // Determine target port count
      // If edges exist: max connected port + 1 (for next connection)
      // If no edges: min_count only
      const targetPortCount = maxPortNumber > 0 ? maxPortNumber + 1 : min_count;

      console.log(`[Dynamic Ports] Calculated:`, {
        maxPortNumber,
        currentMax,
        targetPortCount,
        min_count,
      });

      // Add missing ports if needed
      if (targetPortCount > currentMax) {
        const newInputs = [...node.data.inputs];
        for (let i = currentMax + 1; i <= targetPortCount; i++) {
          // Use old naming (var1, var2) if edges use it, otherwise new naming (var_1, var_2)
          const portName = useOldStyle ? `${prefix}${i}` : `${prefix}_${i}`;

          newInputs.push({
            name: portName,
            port_type: port_type,
            required: false,
            description: `${prefix.charAt(0).toUpperCase() + prefix.slice(1)} ${i}`,
          });
        }
        nodeIdsNeedingUpdate.push(node.id);
        return {
          ...node,
          data: {
            ...node.data,
            inputs: newInputs,
          },
        };
      }

      return node;
    });

    // First set nodes with dynamic ports (without edges)
    set({
      nodes: nodesWithDynamicPorts,
      edges: [],
      selectedNode: null,
      nodeIdsNeedingUpdate: nodeIdsNeedingUpdate,
    });

    // Then set edges after React Flow updates node internals
    // This ensures handles exist before edges are created
    setTimeout(() => {
      set({ edges: enrichedEdges });
      console.log(
        `Loaded workflow: ${enrichedNodes.length} nodes, ${enrichedEdges.length} edges`,
      );
      console.log(
        "Nodes:",
        nodesWithDynamicPorts.map((n) => ({
          id: n.id,
          type: n.data.nodeType,
          inputs: n.data.inputs.map((i) => i.name),
        })),
      );
      console.log("Edges:", enrichedEdges);
    }, 100);
  },

  setDefaultModelIfEmpty: (models) => {
    const { nodes, updateNodeProperty } = get();

    // Find LLM nodes with empty model_name
    const llmNodesWithoutModel = nodes.filter(
      (node) =>
        node.data.nodeType === "simple/llm" &&
        (!node.data.properties.model_name ||
          node.data.properties.model_name === ""),
    );

    if (llmNodesWithoutModel.length === 0) {
      return; // No nodes need updating
    }

    // Get first available model from Ollama
    const modelNames = Object.keys(models);
    if (modelNames.length === 0) {
      console.warn("No Ollama models available");
      return;
    }

    const firstModel = modelNames[0];
    console.log(`Setting default model to: ${firstModel}`);

    // Update all LLM nodes without model
    llmNodesWithoutModel.forEach((node) => {
      updateNodeProperty(node.id, "model_name", firstModel);
    });
  },
  }))
);

export const useNodes = () => useWorkflowStore((s) => s.nodes);
export const useEdges = () => useWorkflowStore((s) => s.edges);
export const useSelectedNode = () => useWorkflowStore((s) => s.selectedNode);
export const useNodeIdsNeedingUpdate = () =>
  useWorkflowStore((s) => s.nodeIdsNeedingUpdate);
export const useSelectedNodeData = () =>
  useWorkflowStore((s) => {
    const node = s.nodes.find((n) => n.id === s.selectedNode);
    return node?.data ?? null;
  });
