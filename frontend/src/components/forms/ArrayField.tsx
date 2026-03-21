import { FormField } from "./FormField";
import type { PropertyDefinition } from "../../types";

interface ArrayFieldProps {
  label: string;
  value: string[];
  definition?: PropertyDefinition;
  onChange: (value: string[]) => void;
}

export function ArrayField({
  label,
  value,
  definition,
  onChange,
}: ArrayFieldProps) {
  // Ensure there's always at least one field (even if empty array)
  const displayValue = value.length === 0 ? [""] : value;

  // Ensure last item is always empty (placeholder for adding new items)
  const itemsToDisplay =
    displayValue[displayValue.length - 1] === ""
      ? displayValue
      : [...displayValue, ""];

  const handleRemove = (index: number) => {
    const newArray = value.filter((_, i) => i !== index);
    onChange(newArray);
  };

  const handleChange = (index: number, newValue: string) => {
    const newArray = [...value];

    // If editing placeholder field (last empty field), add to array
    if (index >= value.length) {
      newArray.push(newValue);
    } else {
      newArray[index] = newValue;
    }

    // Remove empty items except the last one will be auto-added by itemsToDisplay
    const cleaned = newArray.filter((v) => v.trim() !== "");

    onChange(cleaned);
  };

  const isPlaceholder = (index: number) => {
    return index === itemsToDisplay.length - 1;
  };

  return (
    <FormField label={label} description={definition?.description}>
      <div className="array-field">
        {itemsToDisplay.map((item, index) => (
          <div
            key={index}
            className={`array-item${isPlaceholder(index) ? " placeholder" : ""}`}
          >
            <input
              type="text"
              value={item}
              onChange={(e) => handleChange(index, e.target.value)}
              placeholder={
                isPlaceholder(index) ? "Type to add item..." : undefined
              }
            />
            {!isPlaceholder(index) && (
              <button
                type="button"
                onClick={() => handleRemove(index)}
                className="array-remove-btn"
              >
                ✕
              </button>
            )}
          </div>
        ))}
      </div>
    </FormField>
  );
}
