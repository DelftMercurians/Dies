import { Toaster as Sonner } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="dark"
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-bg-elevated group-[.toaster]:text-text-std group-[.toaster]:border-border-muted group-[.toaster]:rounded-none group-[.toaster]:shadow-none",
          description: "group-[.toast]:text-text-dim",
          actionButton:
            "group-[.toast]:bg-accent-cyan group-[.toast]:text-bg-base group-[.toast]:rounded-none",
          cancelButton:
            "group-[.toast]:bg-bg-overlay group-[.toast]:text-text-dim group-[.toast]:rounded-none",
          error:
            "group-[.toaster]:bg-accent-red/20 group-[.toaster]:border-accent-red/50",
          success:
            "group-[.toaster]:bg-accent-green/20 group-[.toaster]:border-accent-green/50",
          warning:
            "group-[.toaster]:bg-accent-amber/20 group-[.toaster]:border-accent-amber/50",
          info: "group-[.toaster]:bg-accent-blue/20 group-[.toaster]:border-accent-blue/50",
        },
      }}
      {...props}
    />
  );
};

export { Toaster };
