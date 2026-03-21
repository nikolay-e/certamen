# Constants

## Node Types (`nodeTypes.ts`)

**CRITICAL**: This file prevents bugs caused by mismatched node type strings.

### The Problem We Prevent

XYFlow requires nodes to have a `type` property that matches a registered component in the `nodeTypes` registry. Previously, we had:

```typescript
// Canvas.tsx
const nodeTypes = {
  workflow: BaseNode,  // ← Registered as "workflow"
};

// workflowStore.ts (BUG!)
const node = {
  type: "workflowNode",  // ← Typo! Should be "workflow"
  data: { ... }
};
```

This caused **all nodes to be non-interactive** because XYFlow couldn't find the component.

### The Solution

1. **Use constants instead of magic strings**:

   ```typescript
   import { NODE_TYPES } from "../constants/nodeTypes";

   const node = {
     type: NODE_TYPES.WORKFLOW,  // ✅ Type-safe, no typos
     data: { ... }
   };
   ```

2. **Runtime validation**:

   ```typescript
   validateNodeType(nodeType); // Throws error if invalid
   ```

3. **Centralized registry**:

   ```typescript
   const nodeTypes = {
     [NODE_TYPES.WORKFLOW]: BaseNode, // Uses constant, not string
   };
   ```

### Adding New Node Types

When adding a new XYFlow node type component:

1. **Add to `NODE_TYPES`**:

   ```typescript
   export const NODE_TYPES = {
     WORKFLOW: "workflow",
     CUSTOM: "custom", // ← New type
   } as const;
   ```

2. **Register in Canvas.tsx**:

   ```typescript
   const nodeTypes = {
     [NODE_TYPES.WORKFLOW]: BaseNode,
     [NODE_TYPES.CUSTOM]: CustomNode, // ← New component
   };
   ```

3. **Use constant everywhere**:

   ```typescript
   const node = { type: NODE_TYPES.CUSTOM, ... };
   ```

### Testing

Run integration tests to ensure node types are correct:

```bash
npm test src/store/__tests__/workflowStore.test.ts
```

Tests verify:

- All nodes have correct XYFlow type
- All nodes have `propertyDefs` for interactivity
- Invalid node types are handled gracefully
