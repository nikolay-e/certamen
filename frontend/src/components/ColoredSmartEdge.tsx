import { memo, useMemo } from "react";
import { useNodes, getSmoothStepPath, BaseEdge } from "@xyflow/react";
import type { EdgeProps, Node } from "@xyflow/react";
import { SmartBezierEdge } from "@tisoap/react-flow-smart-edge";
import { getPortColor } from "../utils/colors";
import type { NodeData } from "../types";

interface ColoredEdgeData {
  portType?: string;
}

function ColoredSmartEdgeComponent(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    source,
    target,
    sourceHandleId,
    selected,
    markerEnd,
    data,
  } = props;

  const nodes = useNodes<Node<NodeData>>();

  const edgeData = data as ColoredEdgeData | undefined;
  const portType = edgeData?.portType;

  const edgeColor = useMemo(() => {
    if (portType) {
      return getPortColor(portType);
    }

    const sourceNode = nodes.find((n) => n.id === source);
    if (sourceNode && sourceHandleId) {
      const sourcePort = sourceNode.data?.outputs?.find(
        (p) => p.name === sourceHandleId,
      );
      if (sourcePort) {
        return getPortColor(sourcePort.port_type);
      }
    }

    return "#9ca3af";
  }, [nodes, source, sourceHandleId, portType]);

  const strokeWidth = selected ? 3 : 2;

  const hasObstacles = useMemo(() => {
    const minX = Math.min(sourceX, targetX);
    const maxX = Math.max(sourceX, targetX);
    const minY = Math.min(sourceY, targetY);
    const maxY = Math.max(sourceY, targetY);

    return nodes.some((node) => {
      if (node.id === source || node.id === target) return false;

      const nodeX = node.position.x;
      const nodeY = node.position.y;
      const nodeWidth = node.measured?.width ?? 180;
      const nodeHeight = node.measured?.height ?? 100;

      const nodeRight = nodeX + nodeWidth;
      const nodeBottom = nodeY + nodeHeight;

      const padding = 20;
      return (
        nodeX - padding < maxX &&
        nodeRight + padding > minX &&
        nodeY - padding < maxY &&
        nodeBottom + padding > minY
      );
    });
  }, [nodes, sourceX, sourceY, targetX, targetY, source, target]);

  if (hasObstacles) {
    return (
      <SmartBezierEdge
        {...props}
        style={{
          stroke: edgeColor,
          strokeWidth,
          filter: selected ? `drop-shadow(0 0 4px ${edgeColor})` : undefined,
        }}
      />
    );
  }

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 8,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: edgeColor,
          strokeWidth,
          filter: selected ? `drop-shadow(0 0 4px ${edgeColor})` : undefined,
        }}
      />
      <path
        d={edgePath}
        style={{
          stroke: "transparent",
          strokeWidth: 20,
          fill: "none",
        }}
      />
    </>
  );
}

export const ColoredSmartEdge = memo(ColoredSmartEdgeComponent);
