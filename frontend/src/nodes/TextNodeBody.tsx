import { useCallback, useEffect, useRef, useMemo, memo } from "react";
import { useWorkflowStore } from "../store/workflowStore";

interface TextNodeBodyProps {
  nodeId: string;
  pages: string[][];
  currentPage: number;
  hidden?: boolean;
}

function adjustTextareaHeight(textarea: HTMLTextAreaElement) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function TextNodeBodyComponent({
  nodeId,
  pages,
  currentPage,
  hidden = false,
}: TextNodeBodyProps) {
  const { updateNodeProperty } = useWorkflowStore();
  const textareasRef = useRef<Map<number, HTMLTextAreaElement>>(new Map());

  const currentFields = useMemo(
    () => pages[currentPage] || [""],
    [pages, currentPage],
  );
  const totalPages = pages.length;

  // Clear all refs on unmount
  useEffect(() => {
    const refs = textareasRef.current;
    return () => {
      refs.clear();
    };
  }, []);

  // Clear stale refs and adjust heights when page or fields change
  useEffect(() => {
    const refs = textareasRef.current;
    // Clear refs for indices that no longer exist
    const validIndices = new Set(currentFields.map((_, i) => i));
    refs.forEach((_, index) => {
      if (!validIndices.has(index)) {
        refs.delete(index);
      }
    });
    // Adjust heights for remaining textareas
    refs.forEach((textarea) => {
      adjustTextareaHeight(textarea);
    });
  }, [currentFields, currentPage]);

  const updatePages = useCallback(
    (newPages: string[][]) => {
      updateNodeProperty(nodeId, "pages", newPages);
    },
    [nodeId, updateNodeProperty],
  );

  const setCurrentPage = useCallback(
    (pageIndex: number) => {
      updateNodeProperty(nodeId, "current_page", pageIndex);
    },
    [nodeId, updateNodeProperty],
  );

  const handleFieldChange = useCallback(
    (fieldIndex: number, value: string) => {
      const newPages = [...pages];
      const newFields = [...(newPages[currentPage] || [""])];
      newFields[fieldIndex] = value;

      // Auto-add empty field at end when typing in last field
      const lastField = newFields[newFields.length - 1];
      if (lastField !== "") {
        newFields.push("");
      }

      // Remove trailing empty fields, but keep at least one
      while (
        newFields.length > 1 &&
        newFields[newFields.length - 1] === "" &&
        newFields[newFields.length - 2] === ""
      ) {
        newFields.pop();
      }

      newPages[currentPage] = newFields;
      updatePages(newPages);
    },
    [pages, currentPage, updatePages],
  );

  const addNewPage = useCallback(() => {
    const newPages = [...pages, [""]];
    updatePages(newPages);
    setCurrentPage(newPages.length - 1);
  }, [pages, updatePages, setCurrentPage]);

  const goToPage = useCallback(
    (index: number) => {
      if (index >= 0 && index < totalPages) {
        setCurrentPage(index);
      }
    },
    [totalPages, setCurrentPage],
  );

  const deleteField = useCallback(
    (fieldIndex: number) => {
      if (currentFields.length <= 1) return;

      const newPages = [...pages];
      const newFields = [...(newPages[currentPage] || [""])];
      newFields.splice(fieldIndex, 1);

      // Ensure at least one empty field remains
      if (newFields.every((f) => f !== "")) {
        newFields.push("");
      }

      newPages[currentPage] = newFields;
      updatePages(newPages);
    },
    [pages, currentPage, currentFields.length, updatePages],
  );

  return (
    <div className="text-node-body nodrag">
      {/* Header with + Page button */}
      <div className="text-node-header">
        <span className="text-node-title">
          {totalPages > 1 ? `Page ${currentPage + 1}/${totalPages}` : "Texts"}
        </span>
        <button
          className="text-node-add-page nodrag"
          onClick={(e) => {
            e.stopPropagation();
            addNewPage();
          }}
          title="Add new page"
        >
          + Page
        </button>
      </div>

      {/* Fields Container */}
      <div className="text-node-fields">
        {currentFields.map((value, index) => (
          <div key={index} className="text-node-field">
            <span className="text-node-field-number">{index + 1}</span>
            <div className="text-node-field-input-wrapper">
              <textarea
                ref={(el) => {
                  if (el) {
                    textareasRef.current.set(index, el);
                    adjustTextareaHeight(el);
                  } else {
                    textareasRef.current.delete(index);
                  }
                }}
                className="text-node-textarea nodrag nowheel"
                value={hidden ? "••••••••" : value}
                onChange={(e) => {
                  e.stopPropagation();
                  if (!hidden) {
                    handleFieldChange(index, e.target.value);
                    adjustTextareaHeight(e.target);
                  }
                }}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                placeholder={
                  index === currentFields.length - 1 ? "Type to add..." : ""
                }
                disabled={hidden}
                rows={1}
              />
              {value && currentFields.length > 1 && !hidden && (
                <button
                  className="text-node-delete-field nodrag"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteField(index);
                  }}
                  title="Delete field"
                >
                  ×
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="text-node-pagination nodrag">
          <button
            className="text-node-nav-btn nodrag"
            onClick={(e) => {
              e.stopPropagation();
              goToPage(currentPage - 1);
            }}
            disabled={currentPage === 0}
          >
            ‹
          </button>

          <div className="text-node-page-buttons">
            {pages.map((_, index) => (
              <button
                key={index}
                className={`text-node-page-btn nodrag ${index === currentPage ? "active" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  goToPage(index);
                }}
              >
                {index + 1}
              </button>
            ))}
          </div>

          <button
            className="text-node-nav-btn nodrag"
            onClick={(e) => {
              e.stopPropagation();
              goToPage(currentPage + 1);
            }}
            disabled={currentPage === totalPages - 1}
          >
            ›
          </button>
        </div>
      )}
    </div>
  );
}

export const TextNodeBody = memo(TextNodeBodyComponent);
