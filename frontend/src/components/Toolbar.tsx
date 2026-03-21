import { APP_VERSION } from "../lib/version";

interface ToolbarProps {
  onExecute: () => void;
  onCancel: () => void;
  onClear: () => void;
  onLoadWorkflow: () => void;
  onSaveWorkflow: () => void;
  executing: boolean;
  hasNodes: boolean;
}

export function Toolbar({
  onExecute,
  onCancel,
  onClear,
  onLoadWorkflow,
  onSaveWorkflow,
  executing,
  hasNodes,
}: ToolbarProps) {
  return (
    <div className="toolbar">
      <div className="toolbar-left">
        <h1 className="toolbar-title">Certamen Workflow Editor</h1>
        <span className="toolbar-version">v{APP_VERSION}</span>
      </div>
      <div className="toolbar-right">
        <button
          className="toolbar-button secondary"
          onClick={onLoadWorkflow}
          disabled={executing}
        >
          Load
        </button>
        <button
          className="toolbar-button secondary"
          onClick={onSaveWorkflow}
          disabled={!hasNodes || executing}
        >
          Save
        </button>
        <button
          className="toolbar-button secondary"
          onClick={onClear}
          disabled={!hasNodes || executing}
        >
          Clear
        </button>
        {executing ? (
          <button className="toolbar-button danger" onClick={onCancel}>
            Stop
          </button>
        ) : (
          <button
            className="toolbar-button primary"
            onClick={onExecute}
            disabled={!hasNodes}
          >
            Execute
          </button>
        )}
      </div>
    </div>
  );
}
