import { renderHook } from "@testing-library/react";
import type { Node } from "@xyflow/react";
import { beforeEach, describe, expect, it } from "vitest";
import { validateNodeType } from "../constants/nodeTypes";
import type { NodeData } from "../types";
import { useNodes, useWorkflowStore } from "./workflowStore";

function makeNode(id: string): Node<NodeData> {
  return {
    id,
    type: "workflow",
    position: { x: 0, y: 0 },
    data: {
      label: id,
      nodeType: "simple/text",
      category: "Input",
      inputs: [],
      outputs: [],
      properties: {},
      propertyDefs: {},
    },
  };
}

describe("workflowStore", () => {
  beforeEach(() => {
    useWorkflowStore.getState().clearWorkflow();
  });

  it("round-trips real store state through addNode and deleteNode", () => {
    useWorkflowStore.getState().addNode(makeNode("n1"));
    expect(useWorkflowStore.getState().nodes.map((n) => n.id)).toContain("n1");

    useWorkflowStore.getState().deleteNode("n1");
    expect(useWorkflowStore.getState().nodes.map((n) => n.id)).not.toContain("n1");
  });

  it("exposes added nodes through the useNodes selector hook", () => {
    useWorkflowStore.getState().addNode(makeNode("n2"));
    const { result } = renderHook(() => useNodes());
    expect(result.current.map((n) => n.id)).toContain("n2");
  });
});

describe("validateNodeType", () => {
  it("accepts a registered node type", () => {
    expect(() => validateNodeType("workflow")).not.toThrow();
  });

  it("throws on an unknown node type", () => {
    expect(() => validateNodeType("bogus")).toThrow(/Invalid node type/);
  });
});
