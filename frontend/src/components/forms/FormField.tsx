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
}: Readonly<FormFieldProps>) {
  return (
    <div className="property-field">
      {/* biome-ignore lint/a11y/noLabelWithoutControl: the field control is supplied via children */}
      <label title={description}>
        {formatLabel(label)}
        {required && <span className="required">*</span>}
      </label>
      {children}
    </div>
  );
}
