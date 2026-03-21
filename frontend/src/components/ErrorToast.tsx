import { useEffect, useState } from "react";
import type { ErrorInfo } from "../types";
import { mapTechnicalError } from "../utils/errorMessages";

interface ErrorToastProps {
  error: ErrorInfo | null;
  onClose: () => void;
}

export function ErrorToast({ error, onClose }: ErrorToastProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (error) {
      // Auto-show toast when error appears - setState is intentional for UI animation
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setVisible(true);
      const timer = setTimeout(() => {
        setVisible(false);
        setTimeout(onClose, 300);
      }, 10000);
      return () => clearTimeout(timer);
    }
  }, [error, onClose]);

  if (!error) return null;

  const mappedError = mapTechnicalError(error.type, error.message);

  const handleActionClick = () => {
    if (mappedError.action === "Retry") {
      // Trigger retry event or reload
      window.location.reload();
    } else if (mappedError.action === "Check credentials") {
      // Navigate to settings or show credentials panel
      console.log("Navigate to settings");
    }
  };

  return (
    <div className={`error-toast ${visible ? "visible" : ""}`}>
      <div className="error-toast-header">
        <span className="error-toast-title">{mappedError.title}</span>
        <button
          className="error-toast-close"
          onClick={() => {
            setVisible(false);
            setTimeout(onClose, 300);
          }}
        >
          ✕
        </button>
      </div>
      <div className="error-toast-body">
        <div className="error-toast-message">{mappedError.message}</div>

        <div className="error-toast-suggestions">
          <div className="error-toast-suggestions-title">What to do:</div>
          <ul className="error-toast-suggestions-list">
            {mappedError.suggestions.map((suggestion, index) => (
              <li key={index}>{suggestion}</li>
            ))}
          </ul>
        </div>

        {mappedError.action && (
          <button
            className="error-toast-action-button"
            onClick={handleActionClick}
          >
            {mappedError.action}
          </button>
        )}

        {error.node_id && (
          <div className="error-toast-node">Node: {error.node_id}</div>
        )}

        {error.traceback && (
          <details className="error-toast-traceback">
            <summary>Show technical details</summary>
            <pre>{error.traceback}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
