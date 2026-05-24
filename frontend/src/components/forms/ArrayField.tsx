import type { PropertyDefinition } from "../../types";
import { FormField } from "./FormField";

interface ArrayFieldProps {
  label: string;
  value: string[];
  definition?: PropertyDefinition;
  onChange: (value: string[]) => void;
}

export function ArrayField({ label, value, definition, onChange }: Readonly<ArrayFieldProps>) {
  const displayValue = value.length === 0 ? [""] : value;

  const itemsToDisplay = displayValue.at(-1) === "" ? displayValue : [...displayValue, ""];

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
            // biome-ignore lint/suspicious/noArrayIndexKey: string[] items have no stable id
            key={index}
            className={`array-item${isPlaceholder(index) ? " placeholder" : ""}`}
          >
            <input
              type="text"
              value={item}
              onChange={(e) => handleChange(index, e.target.value)}
              placeholder={isPlaceholder(index) ? "Type to add item..." : undefined}
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
