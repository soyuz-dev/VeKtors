# VeKtors v0.3-alpha

A small, Kotlin-first numerical computing library for vectors, matrices, tensors, and related numerical operations.

VeKtors is designed for expressive numerical code in areas such as simulations, graphics, games, and scientific computing, while keeping the API idiomatic to Kotlin and the implementation relatively small and understandable.

## Current Status

VeKtors is in ***extremely* early development**. The API is currently being built from the ground up, so breaking changes are expected.

Current published release: **v0.2-alpha**  
Next release: **v0.3-alpha (in development)**

The upcoming release expands the matrix API with factorisation, linear-system solving, rank, and row reduction. The library currently uses `Float` for numerical storage and computation; numerical accuracy on ill-conditioned problems is an ongoing concern.

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

- Elementwise and scalar arithmetic
- Matrix multiplication and matrix-vector multiplication
- Transposition, with `.T` as shorthand
- Identity and diagonal matrix factories
- Trace, determinant, and inverse
- Direct linear-system solving
- LU factorisation with partial pivoting
- Rank for square and rectangular matrices
- Reduced row echelon form (RREF), including pivot and free columns

#### Construction and elementary operations

```kotlin
val a = Matrix(
    floatArrayOf(2f, 1f),
    floatArrayOf(1f, 3f),
)

val identity = Matrix.identity(3)
val diagonal = Matrix.diagonal(2f, 4f, 8f)

val transpose = a.T
val trace = a.trace()
val determinant = a.determinant()
val inverse = a.inverse()
```

Matrices can also be constructed from dimensions and a coordinate-based initializer:

```kotlin
val matrix = Matrix(2, 3) { row, column ->
    (row * 10 + column).toFloat()
}
```

#### Solving linear systems

Solve `Ax = b` directly without explicitly constructing an inverse:

```kotlin
val a = Matrix(
    floatArrayOf(2f, 1f),
    floatArrayOf(1f, 3f),
)
val b = Vector(5f, 7f)

val x = a.solve(b)
// Approximately [1.6, 1.8]
```

`solve()` also accepts a `MatrixLike` right-hand side to solve multiple systems at once. Square nonsingular coefficient matrices are required.

#### LU factorisation: `Pluto`

VeKtors implements LU factorisation with partial pivoting:

\[
PA = LU
\]

```kotlin
val pluto = a.lu

val x = pluto.solve(b)
val determinant = pluto.determinant
val inverse = pluto.inverse

val lower = pluto.L
val upper = pluto.U
val permutation = pluto.P
val pivotOrder = pluto.permutation
```

`a.lu` is a **computed property**: each access builds a fresh factorisation. Store the returned `Pluto` instance if you need to solve multiple systems with the same coefficient matrix.

```kotlin
val pluto = a.lu
val first = pluto.solve(Vector(5f, 7f))
val second = pluto.solve(Vector(1f, 4f))
```

The factorisation owns a snapshot of its input, so subsequent edits to a `MutableMatrix` do not affect an existing `Pluto` instance. `L` and `U` are extractable for nonsingular matrices; `P` represents the row permutation. The decomposition can be checked approximately with `pluto.P * a` and `pluto.L * pluto.U`.

#### Rank and row reduction

`rank()` supports both square and rectangular matrices:

```kotlin
val dependent = Matrix(
    floatArrayOf(1f, 2f, 3f),
    floatArrayOf(2f, 4f, 6f),
)

val rank = dependent.rank()
// 1
```

Use `rref()` to obtain a reduced matrix along with pivot and free column indices:

```kotlin
val result = dependent.rref()

val reduced = result.matrix
val pivots = result.pivotColumns
val free = result.freeColumns
val rankFromReduction = result.rank
```

Both `rank()` and `rref()` accept an optional tolerance (default `1e-6f`) for numerical pivot detection. The tolerance is applied relative to the matrix's largest absolute entry. Results for nearly dependent rows can depend on the tolerance and floating-point precision; they should not be treated as exact symbolic results.

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

- Elementwise and scalar arithmetic
- Broadcasting
- Mapping and reductions
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

For a tensor with shape `[2, 3, 4]`, this produces shape `[4, 2, 3]` while preserving the underlying values.

Because `T` is defined on `TensorLike`, the same interface is available across immutable and mutable tensor implementations. Mutable permutations remain writable views over the original storage.

### Conversions

Convert between vectors, matrices, and tensors when their ranks are compatible:

```kotlin
val tensorFromVector = Vector(1f, 2f, 3f).toTensor()
val tensorFromMatrix = Matrix.identity(3).toTensor()

val vector = tensorFromVector.toVector()
val matrix = tensorFromMatrix.toMatrix()
```

Converting a tensor to a vector requires rank 1; converting to a matrix requires rank 2. Conversions create values of the requested type rather than exposing a mutable alias to the source.

## Immutable and Mutable Types

VeKtors keeps immutable and mutable numerical types separate.

```text
VectorLike        MatrixLike        TensorLike
    │                  │                 │
 Vector             Matrix             Tensor
    │                  │                 │
MutableVector     MutableMatrix     MutableTensor
```

The `*Like` interfaces define common read-only numerical behaviour shared by their concrete implementations. The concrete immutable and mutable types are separate implementations, not subclasses of one another.

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

The current tensor implementation uses flat `FloatArray` storage with shape, stride, and offset metadata. Matrix factorisation instead makes an independent working copy, preserving the source matrix.

## Testing

VeKtors includes JUnit tests covering:

- Tensor construction, scalar tensors, and indexing
- Reshaping, broadcasting, and elementwise operations
- Mutable operations and shared backing storage
- Strided slicing and mixed `Index`, `Range`, `Slice`, and `All` indexing
- Axis permutation and slicing after permutation
- Vector/matrix/tensor conversions
- Matrix factories, trace, determinants, and inverses
- Linear-system solving and multiple right-hand sides
- LU factorisation, pivoting, and factor reconstruction
- Matrix rank, numerical tolerances, and RREF

Tests are organised into focused files instead of a single `TensorTest.kt`. The suite is under active expansion; run the repository's Gradle tests for the current result.

## Roadmap

Near-term priorities for **v0.3-alpha**:

- Finish regression testing and documentation for the expanded matrix API
- Review numerical stability and tolerance handling
- Stabilise the public API ahead of the release

Potential **v0.4-alpha** work:

- Householder QR decomposition
- Least-squares solving for overdetermined systems
- Matrix norms and further orthogonality operations
- Null spaces and other row-reduction-based utilities

Longer-term work may include tensor contraction, batched operations, axis-wise reductions, performance profiling, and SIMD where measurements justify it.

The API is still under active development, so these plans may change as the library evolves.

## License

VeKtors is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.

You may use VeKtors in open-source or proprietary projects. Changes made to MPL-covered source files must remain available under the MPL when distributed.

See [`LICENSE`](LICENSE) for the full licence text.

---

And yes, the axis-permutation helper `T` is called `Tea`.

So naturally:

```kotlin
tensor.T[2, 0, 1]
```

is how VeKtors serves Tea.

The LU factorisation helper is called `Pluto`. It handles `P`, `L`, and `U` — because apparently numerical linear algebra now has an astronomy department.
