//! `tunables!` — declare a block of runtime-tunable `f64` knobs with minimal
//! boilerplate.
//!
//! Each entry replaces what used to be a bare `const NAME: f64 = ...;`. The macro
//! turns it into a process-global cell with a live accessor `NAME()` and registers
//! UI metadata (label, help from the doc comment, optional unit/min/max/step). The
//! block also emits three module-level helpers used to discover and drive the
//! knobs from the executor: `__tunable_specs()`, `__tunable_set()`,
//! `__tunable_reset()`.
//!
//! ```ignore
//! tunables! {
//!     section "Approach";
//!
//!     /// Distance at which the fetch commits to its final drive.
//!     #[tunable(unit = "mm", min = 0.0, max = 1000.0, step = 10.0)]
//!     COMMIT_DISTANCE: f64 = 280.0;
//!
//!     DRIBBLER_SPEED: f64 = 0.6;
//! }
//! ```
//!
//! Keys exposed to the UI are namespaced by the defining module's last path
//! segment (e.g. `handle_ball.COMMIT_DISTANCE`), computed at runtime from
//! `module_path!()`, so the same `NAME` may appear in several skills without
//! colliding. The default literal is the single source of truth — a revert clears
//! the override and the cell snaps back to it.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Attribute, Expr, Ident, LitStr, Token, Type,
};

/// One declared knob.
struct Entry {
    name: Ident,
    default: Expr,
    section: Option<String>,
    help: Option<String>,
    unit: Option<Expr>,
    min: Option<Expr>,
    max: Option<Expr>,
    step: Option<Expr>,
}

struct Tunables {
    entries: Vec<Entry>,
}

impl Parse for Tunables {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut entries = Vec::new();
        let mut current_section: Option<String> = None;

        while !input.is_empty() {
            let attrs = Attribute::parse_outer(input)?;
            let name: Ident = input.parse()?;

            // `section "Name";` directive — sets the section for following knobs.
            if attrs.is_empty() && name == "section" && input.peek(LitStr) {
                let s: LitStr = input.parse()?;
                input.parse::<Token![;]>()?;
                current_section = Some(s.value());
                continue;
            }

            // `NAME: f64 = <expr>;`
            input.parse::<Token![:]>()?;
            let _ty: Type = input.parse()?;
            input.parse::<Token![=]>()?;
            let default: Expr = input.parse()?;
            input.parse::<Token![;]>()?;

            let (help, meta) = parse_attrs(&attrs)?;
            entries.push(Entry {
                name,
                default,
                section: current_section.clone(),
                help,
                unit: meta.unit,
                min: meta.min,
                max: meta.max,
                step: meta.step,
            });
        }

        Ok(Tunables { entries })
    }
}

#[derive(Default)]
struct Meta {
    unit: Option<Expr>,
    min: Option<Expr>,
    max: Option<Expr>,
    step: Option<Expr>,
}

/// Extract the joined doc-comment (help text) and the `#[tunable(...)]` metadata
/// from an entry's outer attributes.
fn parse_attrs(attrs: &[Attribute]) -> syn::Result<(Option<String>, Meta)> {
    let mut docs: Vec<String> = Vec::new();
    let mut meta = Meta::default();

    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    docs.push(s.value().trim().to_string());
                }
            }
        } else if attr.path().is_ident("tunable") {
            attr.parse_nested_meta(|nested| {
                let value = nested.value()?.parse::<Expr>()?;
                if nested.path.is_ident("unit") {
                    meta.unit = Some(value);
                } else if nested.path.is_ident("min") {
                    meta.min = Some(value);
                } else if nested.path.is_ident("max") {
                    meta.max = Some(value);
                } else if nested.path.is_ident("step") {
                    meta.step = Some(value);
                } else {
                    return Err(
                        nested.error("unknown tunable attribute (expected unit/min/max/step)")
                    );
                }
                Ok(())
            })?;
        }
    }

    let help = if docs.is_empty() {
        None
    } else {
        Some(docs.join(" ").trim().to_string())
    };
    Ok((help, meta))
}

#[proc_macro]
pub fn tunables(input: TokenStream) -> TokenStream {
    let Tunables { entries } = parse_macro_input!(input as Tunables);

    let mut items = Vec::new();
    let mut spec_exprs = Vec::new();
    let mut set_arms = Vec::new();
    let mut reset_stmts = Vec::new();

    for e in &entries {
        let name = &e.name;
        let name_str = name.to_string();
        let cell = format_ident!("__TUN_{}", name);
        let default = &e.default;

        items.push(quote! {
            #[allow(non_upper_case_globals, dead_code)]
            static #cell: ::dies_core::Tunable = ::dies_core::Tunable::new(#default);
            #[allow(non_snake_case, dead_code)]
            pub fn #name() -> f64 {
                #cell.get()
            }
        });

        // Section: explicit, else default to the module prefix (skill name).
        let section_call = match &e.section {
            Some(s) => quote! { .section(#s) },
            None => quote! { .section(&__prefix) },
        };
        let help_call = e
            .help
            .as_ref()
            .map(|h| quote! { .help(#h) })
            .unwrap_or_default();
        let unit_call = e
            .unit
            .as_ref()
            .map(|u| quote! { .unit(#u) })
            .unwrap_or_default();
        let min_call = e
            .min
            .as_ref()
            .map(|v| quote! { .min(#v) })
            .unwrap_or_default();
        let max_call = e
            .max
            .as_ref()
            .map(|v| quote! { .max(#v) })
            .unwrap_or_default();
        let step_call = e
            .step
            .as_ref()
            .map(|v| quote! { .step(#v) })
            .unwrap_or_default();

        spec_exprs.push(quote! {
            ::dies_core::TunableSpec::new(&__prefix, #name_str, #cell.default())
                #help_call
                #section_call
                #unit_call
                #min_call
                #max_call
                #step_call
        });

        set_arms.push(quote! {
            #name_str => { #cell.set(value); true }
        });

        reset_stmts.push(quote! { #cell.reset(); });
    }

    let expanded = quote! {
        #(#items)*

        /// Auto-generated: UI specs for every knob declared in this module.
        #[allow(dead_code)]
        pub fn __tunable_specs() -> ::std::vec::Vec<::dies_core::TunableSpec> {
            let __prefix = ::dies_core::tunable_module_prefix(module_path!());
            ::std::vec![ #(#spec_exprs),* ]
        }

        /// Auto-generated: set one knob by its namespaced key. Returns `false` if
        /// the key does not belong to this module.
        #[allow(dead_code)]
        pub fn __tunable_set(key: &str, value: f64) -> bool {
            let __prefix = ::dies_core::tunable_module_prefix(module_path!());
            let bare = match key
                .strip_prefix(__prefix.as_str())
                .and_then(|s| s.strip_prefix('.'))
            {
                ::std::option::Option::Some(b) => b,
                ::std::option::Option::None => return false,
            };
            match bare {
                #(#set_arms)*
                _ => false,
            }
        }

        /// Auto-generated: reset every knob in this module to its code default.
        #[allow(dead_code)]
        pub fn __tunable_reset() {
            #(#reset_stmts)*
        }
    };

    expanded.into()
}
