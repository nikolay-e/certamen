import type { PortDefinition } from "../types";

const TEXT_COMPATIBLE_TYPES = new Set<string>([
  "string",
  "string_matrix",
  "responses",
  "insights",
  "results",
]);

export function isPortCompatible(
  sourcePort: PortDefinition,
  targetPort: PortDefinition,
): boolean {
  const sourceType = sourcePort.port_type;
  const targetType = targetPort.port_type;

  if (sourceType === targetType) {
    return true;
  }

  if (sourceType === "any" || targetType === "any") {
    return true;
  }

  if (TEXT_COMPATIBLE_TYPES.has(sourceType) && TEXT_COMPATIBLE_TYPES.has(targetType)) {
    return true;
  }

  return false;
}
