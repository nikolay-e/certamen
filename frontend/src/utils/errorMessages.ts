export interface UserFriendlyError {
  title: string;
  message: string;
  suggestions: string[];
  action?: string;
}

type ErrorMatcher = (errorMessage: string) => boolean;

interface ErrorMapping {
  matcher: ErrorMatcher;
  getError: (errorMessage: string, nodeType?: string) => UserFriendlyError;
}

const errorMappings: ErrorMapping[] = [
  {
    matcher: (_msg) => _msg.toLowerCase().includes("timeout"),
    getError: (_msg, nodeType) => ({
      title: "Operation timed out",
      message: `The ${nodeType ? nodeType + " node" : "operation"} took too long to complete and was cancelled.`,
      suggestions: [
        "Check your internet connection stability",
        "Try increasing the timeout in node settings",
        "Retry the execution when system load is lower",
      ],
      action: "Retry",
    }),
  },
  {
    matcher: (_msg) => _msg.toLowerCase().includes("no model selected"),
    getError: (_msg, nodeType) => ({
      title: "Model not selected",
      message: "No LLM model was configured for this node.",
      suggestions: [
        `Click on the ${nodeType || "node"} to select it`,
        'Choose a model from the "Model" dropdown in the properties panel',
        "Ensure you have valid API keys configured for your chosen provider",
      ],
      action: "Configure model",
    }),
  },
  {
    matcher: (_msg) => _msg.toLowerCase().includes("cycle"),
    getError: () => ({
      title: "Circular connection detected",
      message:
        "Your workflow contains a loop where a node connects back to itself, either directly or through other nodes.",
      suggestions: [
        "Review your workflow connections by tracing from top to bottom",
        "Remove the connection that creates the loop",
        "Use dedicated loop nodes if repetition is intentional",
      ],
      action: "Review connections",
    }),
  },
  {
    matcher: (_msg) => _msg.toLowerCase().includes("api key"),
    getError: () => ({
      title: "API key issue",
      message:
        "The API key for one of your LLM providers is missing, invalid, or has expired.",
      suggestions: [
        "Check that your API keys are correctly set in environment variables",
        "Verify the API key has not reached its rate limit or expiration",
        "Test the API key directly with the provider to confirm it works",
      ],
      action: "Check credentials",
    }),
  },
  {
    matcher: (_msg) => _msg.toLowerCase().includes("connection"),
    getError: () => ({
      title: "Connection error",
      message: "Could not connect to the LLM service or external API.",
      suggestions: [
        "Verify your internet connection is stable",
        "Check if the LLM service (Ollama, OpenAI, etc.) is running and accessible",
        "Try connecting to a different LLM provider temporarily",
      ],
      action: "Check connection",
    }),
  },
  {
    matcher: (_msg) => _msg.toLowerCase().includes("validation"),
    getError: () => ({
      title: "Validation error",
      message: "The workflow configuration is invalid.",
      suggestions: [
        "Ensure all required node properties are filled in",
        "Check that input and output types match across connections",
        "Review any error messages shown on the nodes themselves",
      ],
      action: "Fix configuration",
    }),
  },
  {
    matcher: (_msg) =>
      _msg.toLowerCase().includes("memory") ||
      _msg.toLowerCase().includes("out of memory"),
    getError: () => ({
      title: "Out of memory",
      message:
        "The system ran out of available memory while processing your workflow.",
      suggestions: [
        "Try running a simpler workflow with fewer nodes",
        "Reduce the size of input data passed to nodes",
        "Close other applications to free up system memory",
      ],
      action: "Simplify workflow",
    }),
  },
  {
    matcher: (_msg) =>
      _msg.toLowerCase().includes("permission") ||
      _msg.toLowerCase().includes("unauthorized"),
    getError: () => ({
      title: "Permission denied",
      message: "You do not have permission to perform this action.",
      suggestions: [
        "Check that you are logged in with the correct account",
        "Verify you have the necessary permissions for this workflow",
        "Contact your system administrator if access should be granted",
      ],
      action: "Check permissions",
    }),
  },
  {
    matcher: (_msg) =>
      _msg.toLowerCase().includes("invalid json") ||
      _msg.toLowerCase().includes("parse error"),
    getError: () => ({
      title: "Invalid data format",
      message: "The data passed to this node could not be parsed correctly.",
      suggestions: [
        "Verify the input data is in the expected format (JSON, text, etc.)",
        "Check for special characters or formatting issues in the input",
        "Use a text validation tool to ensure the data is well-formed",
      ],
      action: "Check input format",
    }),
  },
];

export function mapTechnicalError(
  _errorType: string,
  errorMessage: string,
  nodeType?: string,
): UserFriendlyError {
  for (const mapping of errorMappings) {
    if (mapping.matcher(errorMessage)) {
      return mapping.getError(errorMessage, nodeType);
    }
  }

  return {
    title: "Unexpected error",
    message: `An unexpected error occurred${nodeType ? ` in the ${nodeType} node` : ""}: ${sanitizeErrorMessage(errorMessage)}`,
    suggestions: [
      "Check the browser console for more details (F12 or Cmd+Option+I)",
      "Try refreshing the page and rerunning the workflow",
      "Review your workflow configuration for potential issues",
      "Contact support if the problem persists",
    ],
  };
}

function sanitizeErrorMessage(message: string): string {
  return message.length > 200 ? message.substring(0, 200) + "..." : message;
}
