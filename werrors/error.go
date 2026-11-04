// Package werrors defines the normative WeftCodec v1 error classes.
//
// The interoperable contract is the class (WEFTCODEC-FORMAT §11 and
// WEFT-SELECTORS §5), not the human-readable message.
package werrors

import (
	"errors"
	"fmt"
)

// Class is a normative error class name.
type Class string

const (
	Malformed          Class = "Malformed"
	Corrupt            Class = "Corrupt"
	UnsupportedVersion Class = "UnsupportedVersion"
	UnsupportedMode    Class = "UnsupportedMode"
	UnsupportedFeature Class = "UnsupportedFeature"
	LimitExceeded      Class = "LimitExceeded"
	DependencyMissing  Class = "DependencyMissing"
	DependencyDenied   Class = "DependencyDenied"
	DependencyMismatch Class = "DependencyMismatch"
	DependencyCycle    Class = "DependencyCycle"
	NonCanonical       Class = "NonCanonical"
	Io                 Class = "Io"
	Parse              Class = "Parse"
	UnknownSchema      Class = "UnknownSchema"
)

// Error is a typed codec error. It never panics when constructed or formatted.
type Error struct {
	Kind   Class
	Msg    string
	Pos    int
	HasPos bool
}

// New builds an error with an explicit class.
func New(class Class, message string) *Error {
	return &Error{Kind: class, Msg: message}
}

// Newf builds an error with a formatted message.
func Newf(class Class, format string, args ...any) *Error {
	return &Error{Kind: class, Msg: fmt.Sprintf(format, args...)}
}

// At builds an error carrying a byte position (used by the selector parser).
func At(class Class, pos int, message string) *Error {
	return &Error{Kind: class, Msg: message, Pos: pos, HasPos: true}
}

// Atf builds a positioned error with a formatted message.
func Atf(class Class, pos int, format string, args ...any) *Error {
	return &Error{Kind: class, Msg: fmt.Sprintf(format, args...), Pos: pos, HasPos: true}
}

// Class returns the normative class name.
func (e *Error) Class() string { return string(e.Kind) }

// Error implements error.
func (e *Error) Error() string {
	if e.HasPos {
		return fmt.Sprintf("%s: %s (at byte %d)", e.Kind, e.Msg, e.Pos)
	}
	return fmt.Sprintf("%s: %s", e.Kind, e.Msg)
}

// Position reports the byte position when one is present.
func (e *Error) Position() (int, bool) { return e.Pos, e.HasPos }

// WithPosition returns a copy carrying pos.
func (e *Error) WithPosition(pos int) *Error {
	c := *e
	c.Pos, c.HasPos = pos, true
	return &c
}

// Context prefixes the message.
func (e *Error) Context(context string) *Error {
	c := *e
	c.Msg = context + ": " + c.Msg
	return &c
}

// ClassOf extracts the normative class from any error in the chain. It returns
// the empty class for errors that are not *Error.
func ClassOf(err error) Class {
	var target *Error
	if errors.As(err, &target) {
		return target.Kind
	}
	return ""
}

// Convenience constructors.

func NewMalformed(message string) *Error { return New(Malformed, message) }
func NewCorrupt(message string) *Error   { return New(Corrupt, message) }
func NewUnsupportedVersion(message string) *Error {
	return New(UnsupportedVersion, message)
}
func NewUnsupportedMode(message string) *Error {
	return New(UnsupportedMode, message)
}
func NewUnsupportedFeature(message string) *Error {
	return New(UnsupportedFeature, message)
}
func NewLimitExceeded(message string) *Error {
	return New(LimitExceeded, message)
}
func NewDependencyMissing(message string) *Error {
	return New(DependencyMissing, message)
}
func NewDependencyDenied(message string) *Error {
	return New(DependencyDenied, message)
}
func NewDependencyMismatch(message string) *Error {
	return New(DependencyMismatch, message)
}
func NewDependencyCycle(message string) *Error {
	return New(DependencyCycle, message)
}
func NewNonCanonical(message string) *Error { return New(NonCanonical, message) }
func NewIo(message string) *Error           { return New(Io, message) }
func NewParse(message string) *Error        { return New(Parse, message) }
func NewUnknownSchema(message string) *Error {
	return New(UnknownSchema, message)
}

func Malformedf(format string, args ...any) *Error { return Newf(Malformed, format, args...) }
func Corruptf(format string, args ...any) *Error   { return Newf(Corrupt, format, args...) }
func LimitExceededf(format string, args ...any) *Error {
	return Newf(LimitExceeded, format, args...)
}
func ParseAtf(pos int, format string, args ...any) *Error {
	return Atf(Parse, pos, format, args...)
}
func UnsupportedVersionf(format string, args ...any) *Error {
	return Newf(UnsupportedVersion, format, args...)
}
func UnsupportedModef(format string, args ...any) *Error {
	return Newf(UnsupportedMode, format, args...)
}
func UnsupportedFeaturef(format string, args ...any) *Error {
	return Newf(UnsupportedFeature, format, args...)
}
func DependencyMissingf(format string, args ...any) *Error {
	return Newf(DependencyMissing, format, args...)
}
func DependencyMismatchf(format string, args ...any) *Error {
	return Newf(DependencyMismatch, format, args...)
}
func DependencyCyclef(format string, args ...any) *Error {
	return Newf(DependencyCycle, format, args...)
}
func NonCanonicalf(format string, args ...any) *Error {
	return Newf(NonCanonical, format, args...)
}
func UnknownSchemaf(format string, args ...any) *Error {
	return Newf(UnknownSchema, format, args...)
}
