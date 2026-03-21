import { formatLabel } from "../../utils";

interface FormFieldProps {
  label: string;
  description?: string;
  required?: boolean;
  children: React.ReactNode;
}

export function FormField({
  label,
  description,
  required = false,
  children,
}: FormFieldProps) {
  return (
    <div className="property-field">
      <label title={description}>
        {formatLabel(label)}
        {required && <span className="required">*</span>}
      </label>
      {children}
    </div>
  );
}
