# VeKtors v0.4-alpha

A small, Kotlin-first numerical computing library for vectors, matrices, tensors, and related numerical operations.

VeKtors is designed for expressive numerical code in areas such as simulations, graphics, games, and mathematical experimentation, while keeping the API idiomatic to Kotlin and the implementation relatively small and understandable.

## Current Status

VeKtors is in **early development**. The core numerical types and several linear algebra operations are implemented, but the API is still evolving and breaking changes are possible.

Current published release: **v0.4-alpha**

This release expands VeKtors' numerical linear algebra capabilities with Householder QR decomposition, least-squares solving, matrix norms, and numerical accuracy testing.

VeKtors currently uses `Float` for its public numerical types, with `Double` intermediates in selected numerical algorithms.

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
- QR decomposition using Householder reflections
- Least-squares solving
- Frobenius, one, and infinity norms
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

Matrices can also be constructed using dimensions and a coordinate-based initialiser:

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

`solve()` also accepts a `MatrixLike` right-hand side to solve multiple systems simultaneously.

Direct solving requires a square, nonsingular coefficient matrix.

#### LU factorisation: `Pluto`

VeKtors implements LU factorisation with partial pivoting:

**PA = LU**

```kotlin
val pluto = a.lu

val lower = pluto.L
val upper = pluto.U
val permutation = pluto.P
val pivotOrder = pluto.permutation

val determinant = pluto.determinant
val inverse = pluto.inverse
val solution = pluto.solve(b)
```

The factorisation can be reused to solve multiple systems with the same coefficient matrix:

```kotlin
val pluto = a.lu

val first = pluto.solve(Vector(5f, 7f))
val second = pluto.solve(Vector(1f, 4f))
```

`a.lu` is a computed property: each access creates a new factorisation.

The resulting `Pluto` instance is independent of its source matrix, so subsequent modifications to a `MutableMatrix` do not affect an existing factorisation.

The `L` and `U` factors can be extracted for nonsingular matrices. The `P` matrix represents the row permutations applied during factorisation.

#### QR decomposition: `Barcode`

VeKtors implements QR decomposition using Householder reflections:

**A = QR**

For an `m × n` matrix with `m >= n`, the thin QR decomposition produces:

- `Q`: an `m × n` matrix with orthonormal columns
- `R`: an `n × n` upper-triangular matrix

```kotlin
val a = Matrix(
    floatArrayOf(1f, 2f),
    floatArrayOf(3f, 4f),
    floatArrayOf(5f, 6f),
)

val barcode = a.qr

val q = barcode.Q
val r = barcode.R

val reconstructed = q * r
```

Like `lu`, `qr` is a computed property. Store the returned `Barcode` instance to reuse its factorisation.

Householder reflectors are stored internally, avoiding the need to construct `Q` explicitly during factorisation. Selected intermediate calculations use `Double` precision.

QR decomposition supports square and tall matrices, including rank-deficient matrices. Wide matrices are not currently supported.

#### Least-squares solving

For an overdetermined system, VeKtors can find a solution minimising the squared residual:

**minimise ||Ax − b||²**

```kotlin
val a = Matrix(
    floatArrayOf(1f, 0f),
    floatArrayOf(0f, 1f),
    floatArrayOf(1f, 1f),
)

val b = Vector(1f, 2f, 4f)

val x = a.solveLeastSquares(b)
```

Least-squares solving uses the QR decomposition rather than explicitly forming the normal equations.

The factorisation can be reused:

```kotlin
val barcode = a.qr

val first = barcode.solveLeastSquares(b1)
val second = barcode.solveLeastSquares(b2)
```

The solver currently requires full column rank. Rank-deficient or numerically singular systems are rejected according to a configurable tolerance.

Minimum-norm solutions for underdetermined or rank-deficient systems are not currently supported.

#### Matrix norms

VeKtors provides three standard matrix norms:

```kotlin
val frobenius = a.frobeniusNorm()
val one = a.oneNorm()
val infinity = a.infinityNorm()
```

- **Frobenius norm:** Square root of the sum of squared matrix entries.
- **One norm:** Maximum absolute column sum.
- **Infinity norm:** Maximum absolute row sum.

Norms support square and rectangular matrices.

These operations are also useful for measuring numerical error in factorisations and linear-system solutions.

For example, QR reconstruction error can be estimated using:

```kotlin
val barcode = a.qr
val reconstructed = barcode.Q * barcode.R

val error = (a - reconstructed).frobeniusNorm()
val relativeError = error / a.frobeniusNorm()
```

The relative-error expression assumes a nonzero matrix norm.

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

Both `rank()` and `rref()` accept an optional tolerance, defaulting to `1e-6f`, for numerical pivot detection.

The tolerance is applied relative to the largest absolute matrix entry. Results for nearly dependent rows can depend on the tolerance and floating-point precision.

These are numerical algorithms, not symbolic row-reduction operations.

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

For explicit indexing, VeKtors provides `Index`, `Range`, `Slice`, and `All`.

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

Because `T` is defined on `TensorLike`, the same interface is available across immutable and mutable tensor implementations.

Mutable permutations remain writable views over the original storage.

### Conversions

Convert between vectors, matrices, and tensors when their ranks are compatible:

```kotlin
val tensorFromVector = Vector(1f, 2f, 3f).toTensor()
val tensorFromMatrix = Matrix.identity(3).toTensor()

val vector = tensorFromVector.toVector()
val matrix = tensorFromMatrix.toMatrix()
```

Converting a tensor to a vector requires rank 1; converting to a matrix requires rank 2.

Conversions create values of the requested type rather than exposing a mutable alias to the source.

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

The concrete immutable and mutable types are separate implementations, not subclasses of one another.

Ordinary arithmetic generally produces immutable values, while mutable types provide explicit in-place operations.

```kotlin
mutableVector += other
mutableMatrix *= 2f
mutableTensor *= 1.5f
```

## Design

VeKtors follows a few broad design principles:

- Prefer idiomatic Kotlin APIs over conventions copied directly from other numerical libraries.
- Keep immutable and mutable types distinct.
- Use lightweight read-only interfaces such as `VectorLike`, `MatrixLike`, and `TensorLike`.
- Prefer views to unnecessary copies for tensor slicing and permutation.
- Use primitive storage internally.
- Keep abstractions small until repeated use justifies them.
- Optimise based on measured workloads rather than assumptions.
- Keep the scope focused on numerical algebra.

Tensor storage uses flat `FloatArray` storage with shape, stride, and offset metadata.

Matrix factorisations create independent working copies, preserving their source matrices.

## Numerical Accuracy and Limitations

VeKtors primarily uses single-precision `Float` values. Some numerical algorithms use `Double` intermediates to reduce accumulated rounding error.

Numerical results are subject to floating-point precision and conditioning.

In particular:

- Ill-conditioned matrices can produce inaccurate solutions.
- LU decomposition uses partial pivoting but is not guaranteed to be stable for every matrix.
- QR decomposition uses Householder reflections without column pivoting.
- Numerical rank and RREF depend on tolerance-based pivot selection.
- Least-squares solving currently requires full column rank.
- Exact-zero and tolerance-based singularity checks are not substitutes for general condition-number estimation.

VeKtors is intended for practical numerical algebra, but it should not yet be treated as a substitute for mature, extensively validated numerical backends in precision-critical applications.

## Testing

VeKtors includes JUnit tests covering:

- Tensor construction, scalar tensors, and indexing
- Reshaping, broadcasting, and elementwise operations
- Mutable operations and shared backing storage
- Strided slicing and mixed `Index`, `Range`, `Slice`, and `All` indexing
- Axis permutation and slicing after permutation
- Vector/matrix/tensor conversions
- Matrix factories, trace, determinants, and inverses
- Direct linear-system solving and multiple right-hand sides
- LU factorisation, pivoting, and factor reconstruction
- Matrix rank, numerical tolerances, and RREF
- QR decomposition and orthogonality
- Least-squares solving and residual properties
- Matrix norms
- Numerical reconstruction errors across different matrix scales
- Mutation safety and reusable factorisation snapshots

Tests are organised into focused classes rather than a single monolithic `TensorTest.kt`.

Numerical accuracy tests check mathematical invariants such as:

- `PA ≈ LU`
- `A ≈ QR`
- `QᵀQ ≈ I`
- `Aᵀ(Ax − b) ≈ 0` for least-squares solutions

These tests provide regression coverage but do not establish numerical stability for every possible input.

## Roadmap

VeKtors aims to remain a focused numerical algebra library rather than expand into a general scientific computing framework.

### Core priorities

- Maintain intuitive and consistent vector, matrix, and tensor APIs.
- Improve numerical correctness and floating-point behaviour.
- Expand general-purpose tensor operations, particularly tensor contraction and axis-wise operations.
- Improve performance based on measured workloads.
- Continue documentation, testing, and API refinement.

### Possible future extensions

- Null-space calculations
- Batched matrix operations
- Additional orthogonality utilities
- Eigenvalue and singular-value decompositions where useful
- Double-precision numerical types

These are possibilities rather than commitments.

Numerical integration, differential equation solvers, statistics, optimisation frameworks, and machine learning systems are outside VeKtors' intended core scope.

The long-term objective is a cohesive, reliable numerical algebra toolkit with a reasonably stable public API.

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

The LU factorisation helper is called `Pluto`. It handles `P`, `L`, and `U` — because obviously numerical linear algebra now requires an astronomy department.

And the QR decomposition helper is called `Barcode`.

You can probably work out why.