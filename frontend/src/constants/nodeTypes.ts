/**
 * Node type constants to prevent magic string bugs
 * These MUST match the nodeTypes registry in Canvas.tsx
 */

export const NODE_TYPES = {
  WORKFLOW: "workflow",
} as const;

export type NodeType = (typeof NODE_TYPES)[keyof typeof NODE_TYPES];

/**
 * Validates that a node type is registered
 * @throws Error if node type is invalid
 */
export function validateNodeType(type: string): asserts type is NodeType {
  const validTypes = Object.values(NODE_TYPES);
  if (!validTypes.includes(type as NodeType)) {
    throw new Error(
      `Invalid node type: "${type}". Must be one of: ${validTypes.join(", ")}`,
    );
  }
}
