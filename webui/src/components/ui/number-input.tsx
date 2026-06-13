import * as React from "react";

import { Input, InputProps } from "@/components/ui/input";

/**
 * Numeric input that doesn't fight the user while typing.
 *
 * Behaviour:
 * - Keeps a local string buffer; the committed `value` prop only overwrites it
 *   while the field is NOT focused. So mid-edit states like "", "-", "1." or
 *   ".5" are preserved instead of being snapped back on every keystroke.
 * - Commits on blur or Enter: parses the buffer, calls `onChange` with the
 *   number if finite, otherwise reverts the buffer to the current `value`.
 * - Rendered as `type="text"` + `inputMode="decimal"`, so there are no spinner
 *   arrows and no native min/max validation getting in the way.
 */
export interface NumberInputProps
  extends Omit<InputProps, "value" | "onChange" | "type"> {
  value: number;
  onChange: (value: number) => void;
}

function format(value: number): string {
  return Number.isFinite(value) ? String(value) : "";
}

const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  ({ value, onChange, onFocus, onBlur, onKeyDown, ...props }, ref) => {
    const [text, setText] = React.useState(() => format(value));
    const [focused, setFocused] = React.useState(false);

    // Pull external updates in only when the user isn't actively editing.
    React.useEffect(() => {
      if (!focused) setText(format(value));
    }, [value, focused]);

    const commit = () => {
      const trimmed = text.trim();
      const parsed = trimmed === "" ? NaN : Number(trimmed);
      if (Number.isFinite(parsed)) {
        if (parsed !== value) onChange(parsed);
        setText(format(parsed));
      } else {
        setText(format(value));
      }
    };

    return (
      <Input
        ref={ref}
        type="text"
        inputMode="decimal"
        value={text}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
          setText(e.target.value)
        }
        onFocus={(e: React.FocusEvent<HTMLInputElement>) => {
          setFocused(true);
          onFocus?.(e);
        }}
        onBlur={(e: React.FocusEvent<HTMLInputElement>) => {
          setFocused(false);
          commit();
          onBlur?.(e);
        }}
        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
          if (e.key === "Enter") {
            commit();
            e.currentTarget.blur();
          }
          onKeyDown?.(e);
        }}
        {...props}
      />
    );
  }
);
NumberInput.displayName = "NumberInput";

export { NumberInput };
