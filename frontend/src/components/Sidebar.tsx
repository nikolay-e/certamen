import type { DragEvent } from "react";
import type { NodesByCategory, NodeDefinition } from "../types";
import { getPortColor } from "../utils";

interface SidebarProps {
  nodeDefinitions: NodesByCategory;
  connected: boolean;
}

const categoryIcons: Record<string, string> = {
  Simple: "⚡",
  Tournament: "🏆",
  Flow: "🔀",
  Knowledge: "📚",
};

export function Sidebar({ nodeDefinitions, connected }: SidebarProps) {
  const onDragStart = (event: DragEvent, nodeDefinition: NodeDefinition) => {
    event.dataTransfer.setData(
      "application/reactflow",
      JSON.stringify(nodeDefinition),
    );
    event.dataTransfer.effectAllowed = "move";
  };

  const categories = Object.keys(nodeDefinitions);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Certamen Nodes</h2>
        <div className={`connection-status ${connected ? "connected" : ""}`}>
          {connected ? "Connected" : "Disconnected"}
        </div>
      </div>

      <div className="sidebar-content">
        {categories.length === 0 ? (
          <div className="loading-nodes">Loading nodes...</div>
        ) : (
          categories.map((category) => (
            <div key={category} className="node-category-section">
              <h3 className="category-title">
                {categoryIcons[category] || "📦"} {category}
              </h3>
              <div className="category-nodes">
                {nodeDefinitions[category].map((nodeDef) => (
                  <div
                    key={nodeDef.node_type}
                    className="draggable-node"
                    draggable
                    onDragStart={(e) => onDragStart(e, nodeDef)}
                  >
                    <span className="node-name">
                      {nodeDef.display_name}
                      {nodeDef.description && (
                        <span className="info-icon" title={nodeDef.description}>
                          ?
                        </span>
                      )}
                    </span>
                    <div className="port-markers-group">
                      {(nodeDef.inputs || []).map((port, idx) => (
                        <span
                          key={`in-${idx}`}
                          className="port-marker"
                          style={{
                            backgroundColor: getPortColor(port.port_type),
                          }}
                          title={`${port.name} (${port.port_type})`}
                        />
                      ))}
                      {(nodeDef.inputs || []).length > 0 &&
                        (nodeDef.outputs || []).length > 0 && (
                          <span className="port-arrow">→</span>
                        )}
                      {(nodeDef.outputs || []).map((port, idx) => (
                        <span
                          key={`out-${idx}`}
                          className="port-marker"
                          style={{
                            backgroundColor: getPortColor(port.port_type),
                          }}
                          title={`${port.name} (${port.port_type})`}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
