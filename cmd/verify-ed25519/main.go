// Command verify-ed25519 is the independent Go cross-check for signed
// receipt-bundle blobs. It verifies a detached Ed25519 signature over a
// 32-byte message (the bundle manifest digest) using the same
// crypto/ed25519 path as capability envelopes.
//
// Usage:
//
//	verify-ed25519 <public-key-hex> <message-hex> <signature-hex>
//
// Exit 0 prints "VERIFY OK"; any failure prints "VERIFY FAIL: ..." to stderr
// and exits 1. This tool only verifies; it never reads or prints a secret.
package main

import (
	"encoding/hex"
	"fmt"
	"os"

	"github.com/infinityabundance/WeftCodec-Go/capability"
	"github.com/infinityabundance/WeftCodec-Go/hash"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: verify-ed25519 <public-key-hex> <message-hex> <signature-hex>")
		os.Exit(64)
	}
	publicKeyBytes, err := hex.DecodeString(os.Args[1])
	if err != nil {
		fail("public key is not hexadecimal: %v", err)
	}
	publicKey, err := hash.FromSlice(publicKeyBytes)
	if err != nil {
		fail("public key: %v", err)
	}
	message, err := hex.DecodeString(os.Args[2])
	if err != nil {
		fail("message is not hexadecimal: %v", err)
	}
	signature, err := hex.DecodeString(os.Args[3])
	if err != nil {
		fail("signature is not hexadecimal: %v", err)
	}
	if err := capability.VerifyBytes(publicKey, message, signature); err != nil {
		fail("signature does not verify: %v", err)
	}
	fmt.Println("VERIFY OK")
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "VERIFY FAIL: "+format+"\n", args...)
	os.Exit(1)
}
