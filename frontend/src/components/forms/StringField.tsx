import { FormField } from "./FormField";
import type { PropertyDefinition } from "../../types";

interface StringFieldProps {
  label: string;
  value: string;
  definition?: PropertyDefinition;
  onChange: (value: string) => void;
}

export function StringField({
  label,
  value,
  definition,
  onChange,
}: StringFieldProps) {
  const isMultiline = definition?.multiline || false;

  return (
    <FormField label={label} description={definition?.description}>
      {isMultiline ? (
        <textarea
          value={value || ""}
          onChange={(e) => onChange(e.target.value)}
          rows={4}
        />
      ) : (
        <input
          type="text"
          value={value || ""}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </FormField>
  );
}
