# VeKtors

A small, Kotlin-first numerical computing library for making cool things with software.

VeKtors provides a clean and expressive API for working with vectors, matrices, and tensors in Kotlin. It is intended for graphics, simulations, games, scientific computing, machine learning, and whatever other numerical doohickeys you feel like making.

The goal is simple: make numerical computing feel natural in Kotlin without burying the maths under unnecessary ceremony.

## Current Status

**Current release: `v0.1-alpha`**

VeKtors is in ***extremely* early development**. The API is currently being built from the ground up, so breaking changes are expected.

Despite that, the core numerical types are now functional and usable.

## Features

### Vectors

`Vector` and `MutableVector` provide arbitrary-dimensional vector operations, including:

* Vector arithmetic
* Scalar multiplication and division
* Dot products
* Magnitude and squared magnitude
* Normalisation
* Distance
* Angles
* Projection
* Linear interpolation
* Mapping and basic reductions

Both implement `VectorLike`, allowing APIs to work with mutable and immutable vectors without caring about their storage semantics.

### Matrices

`Matrix` and `MutableMatrix` provide row-major matrix operations, including:

* Matrix arithmetic
* Scalar multiplication and division
* Matrix multiplication
* Matrix-vector multiplication
* Transposition
* `.T`
* Mapping
* In-place operations for mutable matrices

Both implement `MatrixLike`.

### Tensors

`Tensor` and `MutableTensor` provide arbitrary-rank numerical arrays with flat row-major storage.

Current tensor features include:

* Arbitrary tensor shapes and ranks
* Multidimensional indexing
* Elementwise arithmetic
* Scalar arithmetic
* Broadcasting
* Reshaping
* Slicing
* Dimension selection
* Mapping
* Basic reductions
* Mutable and immutable variants

Both implement `TensorLike`.

## Example

```kotlin
val position = MutableVector(0f, 0f)
val target = Vector(10f, 5f)

val direction = (target - position).normalized()

position += direction * 2f
```

Matrices compose naturally:

```kotlin
val rotation =
    rotationY(angle) *
    rotationX(angle * 0.7f) *
    rotationZ(angle * 0.3f)

val transformed = rotation * point
```

And tensors support broadcasting:

```kotlin
val values = Tensor(2, 3) { (row, column) ->
    (row * 10 + column).toFloat()
}

val offsets = Tensor(3) { (column) ->
    (column + 1).toFloat()
}

val result = values + offsets
```

Shapes are exposed as ordinary read-only Kotlin lists:

```kotlin
result.shape
// [2, 3]

result.rank
// 2

result.size
// 6
```

Tensors can also be sliced and reshaped:

```kotlin
val tensor = Tensor(4, 5, 6) { (x, y, z) ->
    (x * 100 + y * 10 + z).toFloat()
}

val sliced = tensor.slice(
    Slice(1, 3),
    Slice(0, 5, 2),
    Slice(2, 6),
)

val reshaped = tensor.reshape(10, 12)
```

## Design

VeKtors follows a few simple principles:

* **Kotlin-first API** — the library should feel like Kotlin, not a Java or Python API translated into Kotlin.
* **Immutable by default** — ordinary operations produce immutable values.
* **Explicit mutability** — mutable variants are available when in-place operations are useful.
* **Simple representations** — numerical data is currently backed primarily by flat `FloatArray`s.
* **Minimal ceremony** — common mathematical expressions should look like the mathematics they represent.
* **Understandable implementation** — optimisation should not come at the cost of turning the library into incomprehensible machinery without evidence that it is necessary.

The core type structure currently looks like:

```text
VectorLike       MatrixLike       TensorLike
    │                 │               │
  Vector            Matrix          Tensor
    │                 │               │
MutableVector    MutableMatrix   MutableTensor
```

The `*Like` interfaces represent read-only mathematical capabilities rather than mutability or storage. This allows, for example, a function accepting `VectorLike` to work naturally with either a `Vector` or a `MutableVector`.

## Performance

VeKtors currently prioritises a straightforward API and implementation over low-level optimisation.

Numerical values use primitive `FloatArray` storage where appropriate, and operations are generally implemented as simple loops suitable for JVM optimisation.

Explicit SIMD and more specialised optimisation may be explored later as larger tensor workloads make meaningful benchmarking possible.

## Roadmap

Possible future additions include:

* Axis-based tensor reductions
* Tensor axis permutation and transposition
* Tensor contraction
* More advanced slicing and indexing
* Additional matrix operations
* Conversion between vectors, matrices, and tensors
* Performance optimisation based on profiling
* SIMD where it provides a measurable benefit

The roadmap is intentionally flexible. VeKtors is still ***extremely* early***.

## Version

`0.1-alpha`

> Apparently 11 commits are enough for a release.