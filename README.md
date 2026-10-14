# VeKtors v0.5-alpha

A small, Kotlin-first numerical computing library for vectors, matrices, tensors, and related numerical operations.

VeKtors is designed for expressive numerical code in areas such as simulations, graphics, games, and mathematical experimentation, while keeping the API idiomatic to Kotlin and the implementation relatively small and understandable.

## Current Status

VeKtors is in **early development**. The core numerical types, practical linear algebra operations, and general tensor algebra operations are implemented, but the API is still evolving and breaking changes are possible.

Current published release: **v0.5-alpha**

This release expands VeKtors' tensor algebra capabilities with general tensor contraction, outer products, batched matrix multiplication, axis-wise reductions, and general axis-wise transformations.

It builds on the numerical linear algebra features introduced in v0.3-alpha and v0.4-alpha.

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

```kotlin
val a = Vector(1f, 2f, 3f)
val b = Vector(4f, 5f, 6f)

val sum = a + b
val dot = a dot b
val magnitude = a.magnitude()
val normalised = a.normalized()
val projection = a.projected(b)
val interpolated = a.lerp(b, 0.5f)
```

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

VeKtors provides arbitrary-rank tensors backed by flat `FloatArray` storage, with shape, stride, and offset metadata.

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

Tensor operations include:

- Elementwise and scalar arithmetic
- Broadcasting
- Elementwise mapping
- Whole-tensor reductions
- Axis-wise reductions
- General axis-wise transformations
- General tensor contraction
- Outer products
- Matrix multiplication with batch broadcasting
- Reshaping
- Scalar tensors
- Indexed views
- Strided slicing
- Axis permutation

#### Broadcasting

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

Broadcasting follows trailing-dimension alignment. Two dimensions are compatible when they are equal or at least one has size 1.

#### Tensor contraction

VeKtors supports general tensor contraction between arbitrary-rank tensors.

Contraction multiplies tensor entries and sums over specified pairs of axes.

```kotlin
val a = Tensor(2, 3) { (i, j) ->
    (i * 3 + j + 1).toFloat()
}

val b = Tensor(3, 4) { (i, j) ->
    (i * 4 + j + 1).toFloat()
}

val result = a.contract(
    b,
    axesA = intArrayOf(1),
    axesB = intArrayOf(0),
)

// Result shape: [2, 4]
```

The axes in `axesA` and `axesB` are paired in their specified order. Each pair must have matching dimension sizes.

The output dimensions consist of the uncontracted axes of the first tensor, followed by those of the second tensor, preserving their relative order.

For example:

```kotlin
val a = Tensor(2, 3, 4) { 1f }
val b = Tensor(3, 4, 5) { 2f }

val result = a.contract(
    b,
    axesA = intArrayOf(1, 2),
    axesB = intArrayOf(0, 1),
)

// Result shape: [2, 5]
```

Contraction supports:

- One or multiple contracted axes
- Arbitrary-rank operands
- Scalar results
- Uncontracted outer products
- Sliced and permuted tensor views
- Immutable and mutable tensor inputs

Contracted axes must be unique within each operand. The number and sizes of paired axes must match.

When all axes are contracted, the result is a rank-zero scalar tensor.

Contraction currently prioritises a general, view-compatible implementation over specialised high-performance kernels.

#### Outer products

The outer product is a contraction with no contracted axes.

VeKtors exposes it directly using the infix `outer` function:

```kotlin
val a = Tensor(3) { (i) ->
    (i + 1).toFloat()
}

val b = Tensor(2) { (i) ->
    (i + 4).toFloat()
}

val result = a outer b

// Shape: [3, 2]
```

For arbitrary-rank tensors, the output shape is the concatenation of both operand shapes.

```kotlin
val a = Tensor(2, 3) { 1f }
val b = Tensor(4, 5) { 2f }

val result = a.outer(b)

// Shape: [2, 3, 4, 5]
```

Both `a outer b` and `a.outer(b)` are supported.

#### Batched matrix multiplication

The infix `matmul` function performs matrix multiplication over the last two dimensions of each operand.

Leading dimensions are interpreted as batch dimensions and broadcast using the same trailing-alignment rules as tensor elementwise operations.

```kotlin
val a = Tensor(10, 3, 4) { (batch, row, column) ->
    (batch + row + column).toFloat()
}

val b = Tensor(10, 4, 5) { (batch, row, column) ->
    (batch + row + column).toFloat()
}

val result = a matmul b

// Shape: [10, 3, 5]
```

Batch dimensions need not be identical if they can be broadcast.

| First tensor | Second tensor | Result |
|---|---|---|
| `[3, 4]` | `[4, 5]` | `[3, 5]` |
| `[10, 3, 4]` | `[10, 4, 5]` | `[10, 3, 5]` |
| `[1, 3, 4]` | `[10, 4, 5]` | `[10, 3, 5]` |
| `[2, 1, 3, 4]` | `[1, 5, 4, 6]` | `[2, 5, 3, 6]` |

Both operands must have rank at least 2, and their inner matrix dimensions must match.

Vector and scalar operands are not currently accepted by `matmul`; general tensor contraction can be used for those cases.

Tensor multiplication distinguishes three operations:

```kotlin
val elementwise = a * b
val matrixProduct = a matmul b
val outerProduct = a outer b
```

`*` remains elementwise multiplication with broadcasting. It is not overloaded to mean matrix multiplication for tensors.

#### Axis-wise reductions

In addition to whole-tensor reductions, VeKtors can reduce individual axes.

```kotlin
val tensor = Tensor(2, 3, 4) { (i, j, k) ->
    (i + j + k).toFloat()
}

val sums = tensor.sum(axis = 1)
val means = tensor.mean(axis = 1)
val minima = tensor.min(axis = 1)
val maxima = tensor.max(axis = 1)

// All result shapes: [2, 4]
```

Axis-wise reduction removes the selected dimension while preserving the relative order of all remaining dimensions.

For a rank-one tensor, reducing its only axis produces a rank-zero scalar tensor.

The parameterless reduction methods remain available:

```kotlin
val total = tensor.sum()
val average = tensor.mean()
val minimum = tensor.min()
val maximum = tensor.max()
```

These aggregate the entire tensor into a `Float`, rather than returning a tensor.

#### General axis-wise transformations

VeKtors provides two general operations for transforming one-dimensional slices along an axis.

`mapAxis()` transforms each slice into another slice of the same length, preserving the original tensor shape.

```kotlin
val tensor = Tensor(2, 3) { (i, j) ->
    (i * 3 + j + 1).toFloat()
}

val centred = tensor.mapAxis(axis = 1) { slice ->
    val average = slice.mean()
    slice.map { it - average }
}

// Shape: [2, 3]
```

The transformation receives a rank-one `TensorLike` and must return another rank-one tensor of the same length.

`reduceAxis()` transforms each slice into a single `Float`, removing the selected axis.

```kotlin
val ranges = tensor.reduceAxis(axis = 1) { slice ->
    slice.max() - slice.min()
}

// Shape: [2]
```

The standard axis-wise `sum`, `mean`, `min`, and `max` operations are built on this general reduction mechanism.

Together, these operations provide three levels of transformation:

| Operation | Input to callback | Output | Shape |
|---|---|---|---|
| `map()` | Scalar value | Scalar value | Preserved |
| `mapAxis()` | One-dimensional slice | One-dimensional slice | Preserved |
| `reduceAxis()` | One-dimensional slice | Scalar value | Selected axis removed |

The axis-wise callbacks run once per slice.

Both operations return independent immutable tensors and support sliced, permuted, and mutable inputs.

The current implementation materialises one-dimensional slices for the callbacks instead of exposing writable views.

#### Reshaping and scalar tensors

Tensors can be reshaped when the total number of elements remains unchanged.

```kotlin
val tensor = Tensor(2, 3, 4) { 1f }

val reshaped = tensor.reshape(6, 4)

// Shape: [6, 4]
```

VeKtors also supports rank-zero tensors.

```kotlin
val scalar = Tensor.scalar(42f)

scalar.shape
// []

scalar.rank
// 0

scalar.valueAt()
// 42f
```

Scalar tensors are useful as the outputs of complete contractions and reductions.

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

General tensor operations, including contraction, batched matrix multiplication, and axis-wise transformations, work with these views through the common `TensorLike` interface.

Ordinary arithmetic and transformation operations produce independent immutable results rather than writable aliases to their inputs.

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

The distinction applies to general tensor algebra as well: operations on mutable tensors produce immutable results unless explicitly documented otherwise.

## Design

VeKtors follows a few broad design principles:

- Prefer idiomatic Kotlin APIs over conventions copied directly from other numerical libraries.
- Keep immutable and mutable types distinct.
- Use lightweight read-only interfaces such as `VectorLike`, `MatrixLike`, and `TensorLike`.
- Prefer views to unnecessary copies for tensor slicing and permutation.
- Use primitive storage internally.
- Express related operations through general abstractions where useful.
- Keep abstractions small until repeated use justifies them.
- Optimise based on measured workloads rather than assumptions.
- Keep the scope focused on numerical algebra.

Tensor storage uses flat `FloatArray` storage with shape, stride, and offset metadata.

General tensor algebra operates through `TensorLike`, allowing the same functions to support contiguous tensors, non-contiguous views, and mutable tensors.

Axis-wise operations share common slice-transformation mechanisms, while outer products reuse general tensor contraction.

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
- Tensor contraction and batched matrix multiplication accumulate in `Double` before converting results to `Float`.
- The precision of reductions depends on their accumulation implementations.
- Operations on extremely large tensors may require substantial intermediate storage.
- General tensor contraction and batched multiplication are not yet backed by specialised high-performance numerical kernels.
- Exact-zero and tolerance-based singularity checks are not substitutes for general condition-number estimation.

Tensor dimensions are currently required to be positive. Empty dimensions and empty tensor slices are not supported.

The current tensor algebra implementations prioritise straightforward, general behaviour over aggressive optimisation.

VeKtors is intended for practical numerical algebra, but it should not yet be treated as a substitute for mature, extensively validated numerical backends in precision-critical applications.

## Testing

VeKtors includes JUnit tests covering:

- Tensor construction, scalar tensors, and indexing
- Reshaping, broadcasting, and elementwise operations
- Mutable operations and shared backing storage
- Strided slicing and mixed `Index`, `Range`, `Slice`, and `All` indexing
- Axis permutation and slicing after permutation
- Vector/matrix/tensor conversions
- Tensor contraction with one or multiple axes
- Complete contraction and scalar results
- Outer products and infix operations
- Batched matrix multiplication
- Batch broadcasting across singleton and differing batch dimensions
- General axis-wise mapping and reduction
- Custom reductions and slice transformations
- Tensor operations on sliced and permuted views
- Invalid contraction axes and incompatible dimensions
- Mutation safety for tensor operation results
- Matrix factories, trace, determinants, and inverses
- Direct linear-system solving and multiple right-hand sides
- LU factorisation, pivoting, and factor reconstruction
- Matrix rank, numerical tolerances, and RREF
- QR decomposition and orthogonality
- Least-squares solving and residual properties
- Matrix norms
- Numerical reconstruction errors across different matrix scales
- Reusable factorisation snapshots

Tests are organised into focused classes rather than a single monolithic `TensorTest.kt`.

Numerical accuracy tests check mathematical invariants such as:

- `PA ≈ LU`
- `A ≈ QR`
- `QᵀQ ≈ I`
- `Aᵀ(Ax − b) ≈ 0` for least-squares solutions

Tensor tests additionally check relationships such as:

- Matrix multiplication through `contract()` agrees with ordinary matrix multiplication.
- Batched `matmul()` agrees with independently calculated matrix products.
- Outer products agree with contraction over zero axes.
- Axis-wise reductions preserve the correct remaining dimensions.
- Operations on tensor views agree with operations on their corresponding logical values.

These tests provide regression coverage but do not establish numerical stability or performance for every possible input.

## Roadmap

VeKtors aims to remain a focused numerical algebra library rather than expand into a general scientific computing framework.

### Core priorities

- Maintain intuitive and consistent vector, matrix, and tensor APIs.
- Improve numerical correctness and floating-point behaviour.
- Benchmark tensor contraction, batched matrix multiplication, and axis-wise transformations.
- Reduce unnecessary intermediate allocations in frequently used operations.
- Improve performance based on measured workloads.
- Continue documentation, testing, and API refinement.
- Progress towards a stable public API.

### Possible future extensions

- Null-space calculations
- Additional orthogonality utilities
- Eigenvalue and singular-value decompositions where useful
- Double-precision numerical types
- Further optimisations for strided tensor operations
- Additional tensor algebra conveniences when justified by real use cases

These are possibilities rather than commitments.

Numerical integration, differential equation solvers, statistics, optimisation frameworks, GPU computing, and machine learning systems are outside VeKtors' intended core scope.

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