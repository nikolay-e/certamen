import { PORT_COLORS, CATEGORY_COLORS } from "../constants";

export function getPortColor(portType: string): string {
  const upperType = portType.toUpperCase();
  return PORT_COLORS[upperType as keyof typeof PORT_COLORS] || PORT_COLORS.ANY;
}

export function getCategoryColor(category: string): string {
  const upperCategory = category.toUpperCase();
  return (
    CATEGORY_COLORS[upperCategory as keyof typeof CATEGORY_COLORS] || "#6b7280"
  );
}
