use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Settings)]
pub fn derive_settings(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let expanded = quote! {
        // Define the static storage
        impl #name {
            fn settings_instance() -> &'static arc_swap::ArcSwap<Self> {
                static SETTINGS: std::cell::OnceCell<arc_swap::ArcSwap<#name>> =
                    std::cell::OnceCell::new();

                SETTINGS.get_or_init(|| {
                    arc_swap::ArcSwap::new(Arc::new(#name::default()))
                })
            }
        }

        // Implement the Settings trait
        impl Settings for #name {
            fn descriptor() -> SettingsDescriptor {
                SettingsDescriptor {
                    type_name: stringify!(#name).to_string(),
                    label: stringify!(#name).to_string(),
                    description: None,
                    fields: vec![], // We'll implement field descriptors later
                }
            }

            fn load() -> Arc<Self> {
                Self::settings_instance().load_full()
            }

            fn store(&self, new_value: Self) {
                Self::settings_instance().store(Arc::new(new_value));
            }
        }
    };

    TokenStream::from(expanded)
}
