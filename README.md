# VeKtors v0.2-alpha

A small, Kotlin-first numerical computing library for vectors, matrices, tensors, and related numerical operations.

VeKtors is designed for expressive numerical code in areas such as simulations, graphics, games, and scientific computing, while keeping the API idiomatic to Kotlin and the implementation relatively small and understandable.

## Current Status

VeKtors is in ***extremely* early development**. The API is currently being built from the ground up, so breaking changes are expected.

Current version: **0.2-alpha**

## Features

### Vectors

Immutable and mutable vectors with support for:

- Elementwise arithmetic
- Scalar arithmetic
- Dot products
- Magnitude and squared magnitude
- Normalisation
- Distance
- Angles
- Projection
- Linear interpolation
- Mapping and reductions

### Matrices

Immutable and mutable matrices with:

- Elementwise arithmetic
- Scalar arithmetic
- Matrix multiplication
- Matrix-vector multiplication
- Transposition

Matrices also provide `.T` as a shorthand for transposition.

### Tensors

VeKtors provides arbitrary-rank tensors backed by flat `FloatArray` storage.

```kotlin
val tensor = Tensor(2, 3, 4) { (x, y, z) ->
    (x * 100 + y * 10 + z).toFloat()
}
```

```kotlin
tensor.shape
// [2, 3, 4]

tensor.rank
// 3

tensor.size
// 24

tensor.valueAt(1, 2, 3)
// 123f
```

Tensor operations currently include:

- Elementwise arithmetic
- Scalar arithmetic
- Broadcasting
- Mapping
- Reductions
- Reshaping
- Scalar tensors
- Indexed views
- Strided slicing
- Axis permutation

### Broadcasting

Tensor elementwise operations support broadcasting between compatible shapes.

```kotlin
val a = Tensor(2, 3) { (row, column) ->
    (row * 10 + column).toFloat()
}

val b = Tensor(3) { (column) ->
    (column + 1).toFloat()
}

val result = a + b
```

Here, the shape `[3]` tensor is broadcast across the leading dimension of the `[2, 3]` tensor.

### Indexing

Simple integer indexing removes the first dimension.

```kotlin
val tensor = Tensor(2, 3, 4) { (x, y, z) ->
    (x * 100 + y * 10 + z).toFloat()
}

tensor[1].shape
// [3, 4]

tensor[1][2].shape
// [4]

tensor[1][2][3].shape
// []

tensor[1][2][3].valueAt()
// 123f
```

For more explicit indexing, VeKtors provides `Index`, `Range`, `Slice`, and `All`.

```kotlin
val view = tensor[
    Index(1),
    Range(0..1),
    Slice(0, 4, 2),
]
```

Their semantics are distinct:

```kotlin
Index(3)
// Select one element and remove the dimension.

Range(2..7)
// Select an inclusive Kotlin range and preserve the dimension.

Slice(2, 7)
// Select [2, 7) and preserve the dimension.

Slice(2, 8, 2)
// Select 2, 4, 6.

All
// Preserve the entire dimension.
```

Missing indices implicitly behave as `All`.

### Views

Tensor indexing, slicing, and axis permutation create views over existing storage rather than copying values where possible.

Mutable views share backing storage with their source tensor.

```kotlin
val tensor = MutableTensor(4, 5, 6) { (x, y, z) ->
    (x * 100 + y * 10 + z).toFloat()
}

val view = tensor[
    Index(2),
    Range(1..3),
    Slice(0, 6, 2),
]

view.setValueAt(
    1,
    2,
    value = 69420f,
)
```

Views are represented using shape, stride, and offset metadata over shared storage.

### Axis Permutation and `T`

`TensorLike` provides the `T` property for axis permutation.

Unlike matrix transposition, tensors do not have a single universal transpose operation. `T` therefore accepts an explicit axis ordering:

```kotlin
val permuted = tensor.T[2, 0, 1]
```

For a tensor with shape:

```text
[2, 3, 4]
```

this produces:

```text
[4, 2, 3]
```

while preserving the underlying values.

Because `T` is defined on `TensorLike`, the same interface is available across immutable and mutable tensor implementations.

Mutable permutations remain writable views over the original storage.

## Immutable and Mutable Types

VeKtors keeps immutable and mutable numerical types separate.

```text
VectorLike        MatrixLike        TensorLike
    │                  │                 │
 Vector             Matrix             Tensor
    │                  │                 │
MutableVector     MutableMatrix     MutableTensor
```

The `*Like` interfaces define common read-only numerical behaviour shared by their concrete implementations.

Ordinary arithmetic generally produces immutable values, while mutable types provide explicit in-place operations.

```kotlin
mutableVector += other
mutableMatrix *= 2f
mutableTensor *= 1.5f
```

## Design

VeKtors currently follows a few broad design principles:

- Prefer idiomatic Kotlin APIs over conventions copied directly from other numerical libraries.
- Keep immutable and mutable types distinct.
- Use lightweight read-only interfaces such as `VectorLike`, `MatrixLike`, and `TensorLike`.
- Prefer views to unnecessary copies for tensor slicing and permutation.
- Use primitive storage internally.
- Keep abstractions small until repeated use justifies them.
- Optimise based on measured workloads rather than assumptions.

The current tensor implementation uses flat `FloatArray` storage with shape, stride, and offset metadata.

## Testing

VeKtors includes JUnit tests covering:

- Tensor construction and indexing
- Scalar tensors
- Reshaping
- Broadcasting
- Elementwise operations
- Mutable operations
- Indexed views
- Strided slicing
- Shared mutable backing storage
- Axis permutation
- Slicing after permutation
- Mixed use of `Index`, `Range`, `Slice`, and `All`

The current test suite contains 32 tests.

## Roadmap

Possible future work includes:

- More tensor operations
- Tensor contractions
- Additional matrix operations
- More reductions
- Improved conversions between vectors, matrices, and tensors
- Further slicing capabilities
- Performance profiling
- SIMD where it is beneficial
- Continued refinement of view semantics and numerical APIs

The API is still under active development, so these plans may change as the library evolves.

## License

VeKtors is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.

You may use VeKtors in open-source or proprietary projects. Changes made to MPL-covered source files must remain available under the MPL when distributed.

See [`LICENSE`](LICENSE) for the full license text.

---

And yes, the axis-permutation helper T is called `Tea`.

So naturally:

```kotlin
tensor.T[2, 0, 1]
```

is how VeKtors serves Tea.