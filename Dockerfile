FROM golang:1.23-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/weftcodec ./cmd/weftcodec

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=build /out/weftcodec /usr/local/bin/weftcodec
LABEL org.opencontainers.image.title="WeftCodec-Go" \
      org.opencontainers.image.description="Independent Go implementation of WeftCodec v1" \
      org.opencontainers.image.source="https://github.com/infinityabundance/WeftCodec-Go" \
      org.opencontainers.image.licenses="Apache-2.0"
ENTRYPOINT ["weftcodec"]
