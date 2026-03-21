// Port type colors - single source of truth
// Models → red
// Text arrays (responses, evaluations) → purple/lilac
// Numeric arrays (scores, rankings) → cyan/teal
export const PORT_COLORS = {
  STRING: "#3b82f6", // blue - single text
  STRING_MATRIX: "#3b82f6", // blue - 2D text array (pages x models)
  MODELS: "#ef4444", // red - array of models
  MODEL: "#f97316", // orange - single model
  RESPONSES: "#a855f7",
  SCORES: "#06b6d4",
  RANKINGS: "#06b6d4",
  RESULTS: "#10b981",
  INSIGHTS: "#c084fc",
  BOOLEAN: "#22c55e",
  NUMBER: "#06b6d4",
  ANY: "#9ca3af",
} as const;

// Category colors
export const CATEGORY_COLORS = {
  SIMPLE: "#3b82f6",
  TOURNAMENT: "#f59e0b",
} as const;
