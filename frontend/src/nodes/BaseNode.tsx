import { memo, useMemo } from "react";
import { Handle, Position, NodeResizer } from "@xyflow/react";
import type { NodeData, PortDefinition } from "../types";
import { useWorkflowStore } from "../store/workflowStore";
import { getPortColor, getCategoryColor } from "../utils";
import { truncateText } from "../utils/formatting";
import { TextNodeBody } from "./TextNodeBody";

interface BaseNodeProps {
  id: string;
  data: NodeData;
  selected?: boolean;
}

const MIN_WIDTH = 180;
const MIN_HEIGHT = 100;

interface TextPreview {
  text: string;
  fullText?: string;
}

function BaseNodeComponent({ id, data, selected }: BaseNodeProps) {
  const { deleteNode } = useWorkflowStore();
  const categoryColor = getCategoryColor(data.category);
  const isExecuting = data.executing;
  const isCompleted = data.completed;
  const hasError = !!data.error;
  const isTextNode = data.nodeType === "simple/text";

  let statusClass = "";
  if (hasError) statusClass = "node-error";
  else if (isCompleted) statusClass = "node-completed";
  else if (isExecuting) statusClass = "node-executing";

  const textPreview = isTextNode ? null : getTextPreview(data);

  // For text nodes: get pages (initialize from texts if needed)
  const textNodeData = useMemo(() => {
    if (!isTextNode) return null;

    const props = data.properties || {};
    let pages = props.pages as string[][] | undefined;
    const texts = props.texts as string[] | undefined;
    const currentPage = Number(props.current_page) || 0;
    const hidden = props.hidden === true;

    // Initialize pages from texts if pages is empty
    if (!pages || !Array.isArray(pages) || pages.length === 0) {
      if (texts && Array.isArray(texts) && texts.length > 0) {
        // Convert texts to single page with auto-add field
        const fieldsWithEmpty = [...texts.filter((t) => t !== undefined)];
        if (
          fieldsWithEmpty.length === 0 ||
          fieldsWithEmpty[fieldsWithEmpty.length - 1] !== ""
        ) {
          fieldsWithEmpty.push("");
        }
        pages = [fieldsWithEmpty];
      } else {
        pages = [[""]];
      }
    }

    return { pages, currentPage, hidden };
  }, [isTextNode, data.properties]);

  const nodeTypeClass = `node-type-${data.nodeType.replace("/", "-")}`;

  return (
    <div
      className={`workflow-node ${nodeTypeClass} ${statusClass} ${selected ? "selected" : ""}`}
      style={{
        borderTopColor: categoryColor,
        minWidth: MIN_WIDTH,
        minHeight: MIN_HEIGHT,
      }}
    >
      <NodeResizer
        minWidth={MIN_WIDTH}
        minHeight={MIN_HEIGHT}
        isVisible={selected}
        lineClassName="node-resize-line"
        handleClassName="node-resize-handle"
      />
      <div className="node-header" style={{ backgroundColor: categoryColor }}>
        <span className="node-category">{data.category}</span>
        <span className="node-title">{data.label}</span>
        {data.hasWarning && (
          <div
            className="node-warning-badge"
            title={
              typeof data.warning === "string"
                ? data.warning
                : "Validation warning"
            }
          >
            ⚠️
          </div>
        )}
        <button
          className="node-delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            deleteNode(id);
          }}
          title="Delete node"
        >
          ✕
        </button>
      </div>

      <div className="node-body">
        {/* Text node body with editable fields */}
        {textNodeData ? (
          <TextNodeBody
            nodeId={id}
            pages={textNodeData.pages}
            currentPage={textNodeData.currentPage}
            hidden={textNodeData.hidden}
          />
        ) : textPreview ? (
          <div className="node-text-preview" title={textPreview.fullText}>
            {textPreview.text}
          </div>
        ) : null}

        {data.inputs.map((input: PortDefinition, index: number) => (
          <div key={input.name} className="node-port input-port">
            <Handle
              type="target"
              position={Position.Left}
              id={input.name}
              style={{
                top: `${30 + index * 24}%`,
                backgroundColor: getPortColor(input.port_type),
              }}
            />
            <span className="port-label">
              {input.name}
              {input.required && <span className="required">*</span>}
              {input.description && (
                <span className="info-icon" title={input.description}>
                  ?
                </span>
              )}
            </span>
          </div>
        ))}

        {data.outputs.map((output: PortDefinition, index: number) => (
          <div key={output.name} className="node-port output-port">
            <span className="port-label">
              {output.name}
              {output.description && (
                <span className="info-icon" title={output.description}>
                  ?
                </span>
              )}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={output.name}
              style={{
                top: `${30 + index * 24}%`,
                backgroundColor: getPortColor(output.port_type),
              }}
            />
          </div>
        ))}

        {hasError && (
          <div className="node-error-message">
            {typeof data.error === "string"
              ? data.error
              : (data.error as unknown as { message?: string })?.message ||
                JSON.stringify(data.error)}
          </div>
        )}
      </div>
    </div>
  );
}

function getTextPreview(data: NodeData): TextPreview | null {
  const props = data.properties || {};
  const result = data.result as Record<string, unknown> | undefined;

  // Text nodes use TextNodeBody component for display
  if (data.nodeType === "simple/text") {
    return null;
  }

  if (props.template !== undefined) {
    if (props.template === "") {
      return {
        text: "(empty template)",
      };
    }
    const fullText = String(props.template);
    const truncated = truncateText(fullText, 100);
    return {
      text: truncated,
      fullText: fullText !== truncated ? fullText : undefined,
    };
  }

  if (props.question !== undefined) {
    if (props.question === "") {
      return {
        text: "(empty question)",
      };
    }
    const fullText = String(props.question);
    const truncated = truncateText(fullText, 100);
    return {
      text: truncated,
      fullText: fullText !== truncated ? fullText : undefined,
    };
  }

  if (props.model !== undefined) {
    return {
      text: `Model: ${props.model}`,
    };
  }

  // For LLM nodes: show model from result (after execution) or properties
  if (data.nodeType === "simple/llm") {
    const modelConfig = result?.model_config as
      | Record<string, unknown>
      | undefined;
    if (modelConfig?.model_name) {
      const name = modelConfig.name ? `${modelConfig.name}: ` : "";
      return {
        text: `${name}${modelConfig.model_name}`,
      };
    }
    if (props.model_name !== undefined && props.model_name !== "") {
      return {
        text: `${props.model_name}`,
      };
    }
    return {
      text: "(no model selected)",
    };
  }

  if (props.model_name !== undefined && props.model_name !== "") {
    return {
      text: `${props.model_name}`,
    };
  }

  // For multi_text nodes, show title
  if (props.title !== undefined) {
    return {
      text: `📋 ${props.title}`,
    };
  }

  return null;
}

export const BaseNode = memo(BaseNodeComponent);
