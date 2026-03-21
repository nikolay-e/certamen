import { useCallback } from "react";
import type { ModelInfo, NodeData, PropertyDefinition } from "../types";
import {
  StringField,
  NumberField,
  BooleanField,
  SelectField,
  ArrayField,
} from "./forms";

interface PropertiesPanelProps {
  node: { id: string; data: NodeData } | null;
  models: Record<string, ModelInfo>;
  onPropertyChange: (nodeId: string, key: string, value: unknown) => void;
}

export function PropertiesPanel({
  node,
  models,
  onPropertyChange,
}: PropertiesPanelProps) {
  if (!node) {
    return (
      <aside className="properties-panel">
        <div className="properties-header">
          <h2>Properties</h2>
        </div>
        <div className="properties-empty">Select a node to edit properties</div>
      </aside>
    );
  }

  const { id, data } = node;

  return (
    <aside className="properties-panel">
      <div className="properties-header">
        <h2>{data.label}</h2>
        <span className="properties-type">{data.nodeType}</span>
      </div>
      <div className="properties-content">
        {Object.entries(data.properties).map(([key, value]) => (
          <PropertyEditor
            key={key}
            nodeId={id}
            propertyKey={key}
            value={value}
            definition={data.propertyDefs?.[key]}
            models={models}
            onChange={onPropertyChange}
            allProperties={data.properties}
          />
        ))}
      </div>
    </aside>
  );
}

interface PropertyEditorProps {
  nodeId: string;
  propertyKey: string;
  value: unknown;
  definition?: PropertyDefinition;
  models: Record<string, ModelInfo>;
  onChange: (nodeId: string, key: string, value: unknown) => void;
  allProperties: Record<string, unknown>;
}

function PropertyEditor({
  nodeId,
  propertyKey,
  value,
  definition,
  models,
  onChange,
  allProperties,
}: PropertyEditorProps) {
  const handleChange = useCallback(
    (newValue: unknown) => {
      onChange(nodeId, propertyKey, newValue);
    },
    [nodeId, propertyKey, onChange],
  );

  // Use definition type if available
  const propType = definition?.type || inferType(value);

  // Select/dropdown (static or dynamic)
  if (propType === "select") {
    const dependentValue = definition?.depends_on
      ? String(allProperties[definition.depends_on])
      : undefined;
    return (
      <SelectField
        label={propertyKey}
        value={value as string}
        definition={definition}
        dependentValue={dependentValue}
        onChange={(v) => handleChange(v)}
      />
    );
  }

  // Array type with options (multi-select checkboxes)
  if (propType === "array" && definition?.options) {
    return (
      <MultiSelectEditor
        label={propertyKey}
        value={value as string[]}
        options={definition.options}
        models={models}
        onChange={handleChange}
      />
    );
  }

  // Array type (dynamic text fields)
  if (propType === "array") {
    return (
      <ArrayField
        label={propertyKey}
        value={value as string[]}
        definition={definition}
        onChange={handleChange}
      />
    );
  }

  // String type
  if (propType === "string" || typeof value === "string") {
    return (
      <StringField
        label={propertyKey}
        value={String(value)}
        definition={definition}
        onChange={(v) => handleChange(v)}
      />
    );
  }

  // Boolean type
  if (propType === "boolean" || typeof value === "boolean") {
    return (
      <BooleanField
        label={propertyKey}
        value={Boolean(value)}
        definition={definition}
        onChange={(v) => handleChange(v)}
      />
    );
  }

  // Number type
  if (propType === "number" || typeof value === "number") {
    return (
      <NumberField
        label={propertyKey}
        value={Number(value)}
        definition={definition}
        onChange={(v) => handleChange(v)}
      />
    );
  }

  // Integer type
  if (propType === "integer") {
    return (
      <NumberField
        label={propertyKey}
        value={Number(value)}
        definition={definition}
        onChange={(v) => handleChange(Math.round(v))}
      />
    );
  }

  // Fallback: show JSON
  return (
    <div className="property-field">
      <label>{propertyKey}</label>
      <span className="property-unknown">{JSON.stringify(value)}</span>
    </div>
  );
}

function inferType(value: unknown): string {
  if (typeof value === "string") return "string";
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  if (Array.isArray(value)) return "array";
  return "unknown";
}

function MultiSelectEditor({
  label,
  value,
  options,
  models,
  onChange,
}: {
  label: string;
  value: string[];
  options: string[];
  models: Record<string, ModelInfo>;
  onChange: (value: string[]) => void;
}) {
  const toggleOption = (opt: string) => {
    if (value.includes(opt)) {
      onChange(value.filter((v) => v !== opt));
    } else {
      onChange([...value, opt]);
    }
  };

  const selectAll = () => onChange([...options]);
  const selectNone = () => onChange([]);

  const formatLabel = (key: string) =>
    key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="property-field model-selector">
      <label>{formatLabel(label)}</label>
      <div className="model-selector-actions">
        <button type="button" onClick={selectAll}>
          All
        </button>
        <button type="button" onClick={selectNone}>
          None
        </button>
        <span className="model-count">
          {value.length}/{options.length}
        </span>
      </div>
      <div className="model-list">
        {options.map((opt) => {
          const modelInfo = models[opt];
          return (
            <label key={opt} className="model-item">
              <input
                type="checkbox"
                checked={value.includes(opt)}
                onChange={() => toggleOption(opt)}
              />
              <span className="model-name">
                {modelInfo?.display_name || opt}
              </span>
              {modelInfo?.provider && (
                <span className="model-provider">{modelInfo.provider}</span>
              )}
            </label>
          );
        })}
      </div>
    </div>
  );
}
