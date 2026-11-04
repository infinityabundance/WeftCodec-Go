// Package dtype defines the v1 tensor element-size table shared by mode-6
// TENSOR frames and the WEFT-TENSOR logical layer.
//
// Keeping the table here avoids an import cycle once mode 10 (TENSOR_SHARDED)
// composes the modes package with the tensor package.
package dtype

// ElementSize returns the fixed little-endian element size for a v1 tensor
// dtype name.
func ElementSize(name string) (uint64, bool) {
	switch name {
	case "f16", "bf16", "i16":
		return 2, true
	case "f32", "i32":
		return 4, true
	case "f64", "i64":
		return 8, true
	case "i8", "u8":
		return 1, true
	default:
		return 0, false
	}
}
