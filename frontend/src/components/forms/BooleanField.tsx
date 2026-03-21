import { FormField } from "./FormField";
import type { PropertyDefinition } from "../../types";

interface BooleanFieldProps {
  label: string;
  value: boolean;
  definition?: PropertyDefinition;
  onChange: (value: boolean) => void;
}

export function BooleanField({
  label,
  value,
  definition,
  onChange,
}: BooleanFieldProps) {
  return (
    <FormField label={label} description={definition?.description}>
      <input
        type="checkbox"
        checked={value || false}
        onChange={(e) => onChange(e.target.checked)}
      />
    </FormField>
  );
}
