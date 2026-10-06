# VeKtors

A small, Kotlin-first numerical computing library for **making cool things with software**.

VeKtors exists to provide a pleasant, expressive API for working with numerical data in Kotlin — from simple vectors and matrices to eventually arbitrary-dimensional tensors. It is intended to be useful for things like graphics, simulations, games, scientific computing, and other projects where mathematical operations should feel natural in code.

The library prioritises a **clean Kotlin API, understandable implementations, and minimal ceremony** over trying to replicate every feature of established numerical-computing libraries.

## Current Status

VeKtors is in ***extremely* early development**. The API is currently being built from the ground up, so breaking changes are expected.

### Implemented

#### `Vector`

The `Vector` API is currently implemented and includes:

* Construction from `Float` values
* Indexed access
* Vector addition and subtraction
* Scalar multiplication and division
* `Float * Vector`
* `sum()`
* `mean()`
* `min()`
* `max()`
* `map()`
* `dot`
* `magnitude()`
* `magnitudeSquared()`
* `normalized()`
* `distanceTo()`
* `angleTo()`
* `projected()`
* `lerp()`

#### `Matrix`

The initial `Matrix` API is implemented alongside `Vector.

It currently includes:

* Construction from rows of `Float`
* Row/column indexing
* Matrix addition and subtraction
* Scalar multiplication and division
* `Float * Matrix`
* Matrix multiplication
* Matrix-vector multiplication
* `transpose()`
* `.T` transpose shorthand

### Planned

#### `Tensor`

`Tensor` is the next major abstraction planned for VeKtors.

It will build on the ideas established by `Vector` and `Matrix`, but **will not inherit from either type**. The goal is to provide an arbitrary-dimensional numerical container with its own shape, indexing, and mathematical operations.

Tensor functionality is not currently part of the public API.

## Design

VeKtors intentionally keeps its abstractions small.

`Vector` serves as a reference implementation for the library's numerical API. `Matrix` extends those ideas into two dimensions while introducing matrix-specific operations.

The eventual `Tensor` abstraction will generalise these concepts to arbitrary dimensions without forcing an inheritance relationship between the numerical types.

## Package

```text
org.soyuz.vektors
```

## Stability

**Pre-alpha / experimental.**

The current API is unstable. Names, signatures, semantics, and internal representations may change substantially as the library develops.
