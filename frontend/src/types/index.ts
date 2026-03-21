import { PortType, NodeCategory, PropertyType, NodeStatus } from "./enums";

// Property values - typed union вместо unknown
export type PropertyValue = string | number | boolean | string[] | null;

// Port definition
export interface PortDefinition {
  name: string;
  port_type: PortType | string; // Allow both enum and string for compatibility
  required: boolean;
  description: string;
}

// Property definition (добавлены missing fields)
export interface PropertyDefinition {
  type: PropertyType; // Enum вместо string
  default: PropertyValue;
  options?: string[];
  description?: string;
  multiline?: boolean;
  min?: number;
  max?: number;
  step?: number;
  dynamic?: boolean;
  depends_on?: string;
  ui_hidden?: boolean; // Hide from Properties panel
  items?: {
    // Для array type
    type: PropertyType;
  };
}

// Dynamic input configuration
export interface DynamicInputConfig {
  prefix: string;
  port_type: string;
  min_count: number;
}

// Node definition (добавлено description field)
export interface NodeDefinition {
  node_type: string; // Соответствует backend
  display_name: string;
  category: NodeCategory; // Enum вместо string
  description: string; // Добавлено
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  properties: Record<string, PropertyDefinition>;
  dynamic_inputs?: DynamicInputConfig;
}

// Node data (используется в Canvas/Store)
// Backward compatible - keeps unknown for properties to allow Phase 2-3 compatibility
export interface NodeData extends Record<string, unknown> {
  label: string;
  nodeType: string;
  category: NodeCategory | string; // Allow both enum and string for compatibility
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  properties: Record<string, unknown>; // Keep unknown for backward compatibility
  propertyDefs: Record<string, PropertyDefinition>;
  dynamicInputs?: DynamicInputConfig; // For nodes with dynamic input ports
  status?: NodeStatus; // Enum
  executing?: boolean; // Legacy - deprecated, use status
  completed?: boolean; // Legacy - deprecated, use status
  error?: string;
  result?: Record<string, unknown>; // Keep unknown for backward compatibility
  hasWarning?: boolean; // Validation warning indicator
  warning?: string; // Validation warning message
}

// Unified error model
export interface ErrorInfo {
  type: string; // Exception class name
  message: string;
  node_id?: string;
  traceback?: string;
}

// WebSocket execution message (used in useWebSocket hook)
export interface ExecutionMessage {
  type: string;
  execution_id?: string;
  node_id?: string;
  node_type?: string;
  outputs?: Record<string, unknown>;
  error?: string;
  data?: Record<string, unknown>;
}

// Model info
export interface ModelInfo {
  display_name: string;
  provider: string;
  model_name?: string;
}

// Nodes by category
export type NodesByCategory = Record<string, NodeDefinition[]>;
