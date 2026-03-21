import { useEffect, useState } from "react";
import { FormField } from "./FormField";
import type { PropertyDefinition } from "../../types";

interface SelectFieldProps {
  label: string;
  value: string;
  definition?: PropertyDefinition;
  dependentValue?: string;
  onChange: (value: string) => void;
}

export function SelectField({
  label,
  value,
  definition,
  dependentValue,
  onChange,
}: SelectFieldProps) {
  const [options, setOptions] = useState<string[]>(definition?.options || []);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!definition?.dynamic || !definition?.depends_on || !dependentValue) {
      setOptions(definition?.options || []);
      return;
    }

    const fetchOptions = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/models/${dependentValue}`);
        const data = await response.json();
        setOptions(data.models || []);
      } catch (error) {
        console.error(`Failed to fetch options for ${label}:`, error);
        setOptions([]);
      } finally {
        setLoading(false);
      }
    };

    fetchOptions();
  }, [
    dependentValue,
    definition?.dynamic,
    definition?.depends_on,
    definition?.options,
    label,
  ]);

  const isDisabled = loading || options.length === 0;

  return (
    <FormField label={label} description={definition?.description}>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={isDisabled}
      >
        {loading ? (
          <option>Loading options...</option>
        ) : options.length === 0 ? (
          <option>No options available</option>
        ) : (
          <>
            {!options.includes(value) && value && (
              <option value={value}>{value}</option>
            )}
            {options.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </>
        )}
      </select>
    </FormField>
  );
}
