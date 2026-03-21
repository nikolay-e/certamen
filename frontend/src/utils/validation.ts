import type { PortDefinition } from "../types";

const TEXT_COMPATIBLE_TYPES = [
  "string",
  "string_matrix",
  "responses",
  "insights",
  "results",
];

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

  // Text-like types are compatible with each other
  if (
    TEXT_COMPATIBLE_TYPES.includes(sourceType) &&
    TEXT_COMPATIBLE_TYPES.includes(targetType)
  ) {
    return true;
  }

  return false;
}
