package org.soyuz.vektors

import kotlin.math.abs

class Pluto private constructor(
    private val data: FloatArray,
    private val pivots: IntArray,
    private val sign: Int,
    val isSingular: Boolean,
) {
    val size: Int
        get() = pivots.size

    val determinant: Float
        get() {
            if (isSingular) return 0f

            var result = sign.toFloat()

            for (i in 0 until size) {
                result *= data[i * size + i]
            }

            return result
        }

    fun solve(b: VectorLike): Vector {
        require(b.size == size) {
            "Vector size ${b.size} must match matrix size $size"
        }
        require(!isSingular) {
            "Matrix is singular and the system has no unique solution"
        }

        val n = size
        val x = FloatArray(n) { b[pivots[it]] }

        // Forward substitution: Ly = Pb
        for (row in 0 until n) {
            for (column in 0 until row) {
                x[row] -= data[row * n + column] * x[column]
            }
        }

        // Back substitution: Ux = y
        for (row in n - 1 downTo 0) {
            for (column in row + 1 until n) {
                x[row] -= data[row * n + column] * x[column]
            }

            x[row] /= data[row * n + row]
        }

        return Vector(*x)
    }

    fun solve(b: MatrixLike): Matrix {
        require(b.rows == size) {
            "Matrix row count ${b.rows} must match matrix size $size"
        }
        require(!isSingular) {
            "Matrix is singular and the system has no unique solution"
        }

        val n = size
        val columns = b.columns

        val x = Array(n) { row ->
            FloatArray(columns) { column ->
                b[pivots[row], column]
            }
        }

        // Forward substitution: LY = PB
        for (row in 0 until n) {
            for (k in 0 until row) {
                val factor = data[row * n + k]

                for (column in 0 until columns) {
                    x[row][column] -= factor * x[k][column]
                }
            }
        }

        // Back substitution: UX = Y
        for (row in n - 1 downTo 0) {
            for (k in row + 1 until n) {
                val factor = data[row * n + k]

                for (column in 0 until columns) {
                    x[row][column] -= factor * x[k][column]
                }
            }

            val diagonal = data[row * n + row]

            for (column in 0 until columns) {
                x[row][column] /= diagonal
            }
        }

        return Matrix(*x)
    }

    val inverse: Matrix
        get() {
            require(!isSingular) {
                "Matrix is singular and cannot be inverted"
            }

            return solve(Matrix.identity(size))
        }

    companion object {
        operator fun invoke(matrix: MatrixLike): Pluto {
            require(matrix.rows == matrix.columns) {
                "LU decomposition requires a square matrix"
            }

            val n = matrix.rows
            val data = FloatArray(n * n) { index ->
                matrix[index / n, index % n]
            }

            val pivots = IntArray(n) { it }
            var sign = 1
            var singular = false

            for (column in 0 until n) {
                var pivot = column

                for (row in column + 1 until n) {
                    if (
                        abs(data[row * n + column]) >
                        abs(data[pivot * n + column])
                    ) {
                        pivot = row
                    }
                }

                if (data[pivot * n + column] == 0f) {
                    singular = true
                    break
                }

                if (pivot != column) {
                    for (k in 0 until n) {
                        val a = column * n + k
                        val b = pivot * n + k

                        val temporary = data[a]
                        data[a] = data[b]
                        data[b] = temporary
                    }

                    val temporary = pivots[column]
                    pivots[column] = pivots[pivot]
                    pivots[pivot] = temporary

                    sign = -sign
                }

                val diagonal = data[column * n + column]

                for (row in column + 1 until n) {
                    val index = row * n + column
                    data[index] /= diagonal

                    val factor = data[index]

                    for (k in column + 1 until n) {
                        data[row * n + k] -=
                            factor * data[column * n + k]
                    }
                }
            }

            return Pluto(data, pivots, sign, singular)
        }
    }
}
