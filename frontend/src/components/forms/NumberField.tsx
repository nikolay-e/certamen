import { FormField } from "./FormField";
import type { PropertyDefinition } from "../../types";

interface NumberFieldProps {
  label: string;
  value: number;
  definition?: PropertyDefinition;
  onChange: (value: number) => void;
}

export function NumberField({
  label,
  value,
  definition,
  onChange,
}: NumberFieldProps) {
  const isInteger = definition?.type === "integer";
  const min = definition?.min;
  const max = definition?.max;
  const step = definition?.step || (isInteger ? 1 : 0.1);

  return (
    <FormField label={label} description={definition?.description}>
      <input
        type="number"
        value={value || 0}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
      />
    </FormField>
  );
}
