FROM rust:1.73.0-alpine3.18

COPY bridge.rs .

RUN rustc bridge.rs && chmod +x bridge

CMD ["./bridge"]