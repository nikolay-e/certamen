import { PORT_COLORS, CATEGORY_COLORS } from "../constants";
import type { PortType, NodeCategory } from "../types/enums";

export function getPortColor(portType: PortType | string): string {
  const upperType = portType.toUpperCase();
  return PORT_COLORS[upperType as keyof typeof PORT_COLORS] || PORT_COLORS.ANY;
}

export function getCategoryColor(category: NodeCategory | string): string {
  const upperCategory = category.toUpperCase();
  return (
    CATEGORY_COLORS[upperCategory as keyof typeof CATEGORY_COLORS] || "#6b7280"
  );
}
