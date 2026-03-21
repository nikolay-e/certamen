// Port types - точное соответствие Python PortType(Enum) в base.py
export const PortType = {
  MODELS: "models",
  MODEL: "model",
  RESPONSES: "responses",
  SCORES: "scores",
  RANKINGS: "rankings",
  RESULTS: "results",
  INSIGHTS: "insights",
  BOOLEAN: "boolean",
  NUMBER: "number",
  STRING: "string",
  STRING_MATRIX: "string_matrix", // 2D array: [[run1_llm1, run1_llm2], [run2_llm1, run2_llm2], ...]
  ANY: "any",
} as const;

export type PortType = (typeof PortType)[keyof typeof PortType];

// Node categories - точное соответствие Python CATEGORY в node classes
export const NodeCategory = {
  SIMPLE: "Simple",
  TOURNAMENT: "Tournament",
} as const;

export type NodeCategory = (typeof NodeCategory)[keyof typeof NodeCategory];

// Property types
export const PropertyType = {
  STRING: "string",
  NUMBER: "number",
  INTEGER: "integer",
  BOOLEAN: "boolean",
  SELECT: "select",
  ARRAY: "array",
} as const;

export type PropertyType = (typeof PropertyType)[keyof typeof PropertyType];

// Node execution status
export const NodeStatus = {
  IDLE: "idle",
  EXECUTING: "executing",
  COMPLETED: "completed",
  ERROR: "error",
} as const;

export type NodeStatus = (typeof NodeStatus)[keyof typeof NodeStatus];
