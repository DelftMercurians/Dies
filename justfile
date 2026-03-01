default_strategy := "concerto"
all_strategies := "concerto test-strategy v0-strategy"

# Build strategy + run dies in simulation mode
dev strategy=default_strategy:
    cargo build -p {{ strategy }}
    cargo run -- --strategy {{ strategy }}

# Build everything (dies + all strategies) in release mode
build:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release -p dies-cli
    for s in {{ all_strategies }}; do
        cargo build --release -p "$s"
    done

# Run dies + vite dev server
webdev strategy=default_strategy:
    cd webui && pnpm run dev &
    cargo build -p {{ strategy }}
    cargo run -- --strategy {{ strategy }}

# Generate TypeScript bindings + build webui
webbuild:
    typeshare . --lang=typescript --output-file=webui/src/bindings.ts
    printf 'export type Vector2 = [number, number];\nexport type Vector3 = [number, number, number];\nexport type Duration = number;\nexport type HashSet<T> = Array<T>;\n' >> webui/src/bindings.ts
    sed -i.bak 's/data?: undefined//g' webui/src/bindings.ts && rm webui/src/bindings.ts.bak
    cd webui && pnpm run build
