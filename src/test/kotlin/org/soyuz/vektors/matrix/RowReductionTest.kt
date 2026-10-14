package org.soyuz.vektors.matrix

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class RowReductionTest {

    @Test
    fun `identity matrix is already reduced`() {
        val result = Matrix.identity(3).rref()

        Assertions.assertEquals(3, result.rank)
        Assertions.assertEquals(listOf(0, 1, 2), result.pivotColumns)
        Assertions.assertEquals(emptyList<Int>(), result.freeColumns)

        for (row in 0 until 3) {
            for (column in 0 until 3) {
                Assertions.assertEquals(
                    if (row == column) 1f else 0f,
                    result.matrix[row, column],
                )
            }
        }
    }

    @Test
    fun `dependent rows reduce correctly`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(2f, 4f, 6f),
            floatArrayOf(1f, 1f, 1f),
        )

        val result = matrix.rref()

        val expected = Matrix(
            floatArrayOf(1f, 0f, -1f),
            floatArrayOf(0f, 1f, 2f),
            floatArrayOf(0f, 0f, 0f),
        )

        for (row in 0 until 3) {
            for (column in 0 until 3) {
                Assertions.assertEquals(
                    expected[row, column],
                    result.matrix[row, column],
                    1e-5f,
                )
            }
        }

        Assertions.assertEquals(listOf(0, 1), result.pivotColumns)
        Assertions.assertEquals(listOf(2), result.freeColumns)
        Assertions.assertEquals(2, result.rank)
    }

    @Test
    fun `wide matrix has free columns`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f, 3f, 4f),
            floatArrayOf(0f, 1f, 2f, 3f),
        )

        val result = matrix.rref()

        Assertions.assertEquals(2, result.rank)
        Assertions.assertEquals(listOf(0, 1), result.pivotColumns)
        Assertions.assertEquals(listOf(2, 3), result.freeColumns)

        Assertions.assertEquals(1f, result.matrix[0, 0], 1e-5f)
        Assertions.assertEquals(0f, result.matrix[0, 1], 1e-5f)
        Assertions.assertEquals(0f, result.matrix[1, 0], 1e-5f)
        Assertions.assertEquals(1f, result.matrix[1, 1], 1e-5f)
    }

    @Test
    fun `tall matrix reduces to identity and zero rows`() {
        val matrix = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val result = matrix.rref()

        Assertions.assertEquals(2, result.rank)
        Assertions.assertEquals(1f, result.matrix[0, 0], 1e-5f)
        Assertions.assertEquals(1f, result.matrix[1, 1], 1e-5f)
        Assertions.assertEquals(0f, result.matrix[2, 0], 1e-5f)
        Assertions.assertEquals(0f, result.matrix[2, 1], 1e-5f)
    }

    @Test
    fun `zero matrix has rank zero`() {
        val matrix = Matrix(3, 4) { _, _ -> 0f }
        val result = matrix.rref()

        Assertions.assertEquals(0, result.rank)
        Assertions.assertEquals(emptyList<Int>(), result.pivotColumns)
        Assertions.assertEquals(listOf(0, 1, 2, 3), result.freeColumns)

        for (row in 0 until 3) {
            for (column in 0 until 4) {
                Assertions.assertEquals(0f, result.matrix[row, column])
            }
        }
    }

    @Test
    fun `reduction skips empty pivot columns`() {
        val matrix = Matrix(
            floatArrayOf(0f, 1f, 2f),
            floatArrayOf(0f, 0f, 1f),
            floatArrayOf(0f, 0f, 0f),
        )

        val result = matrix.rref()

        Assertions.assertEquals(listOf(1, 2), result.pivotColumns)
        Assertions.assertEquals(listOf(0), result.freeColumns)
        Assertions.assertEquals(2, result.rank)

        Assertions.assertEquals(1f, result.matrix[0, 1], 1e-5f)
        Assertions.assertEquals(0f, result.matrix[0, 2], 1e-5f)
        Assertions.assertEquals(1f, result.matrix[1, 2], 1e-5f)
    }

    @Test
    fun `reduction is independent of original mutable matrix`() {
        val matrix = MutableMatrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(2f, 4f),
        )

        val result = matrix.rref()

        matrix[1, 1] = 5f

        Assertions.assertEquals(1, result.rank)
        Assertions.assertEquals(2, matrix.rank())
        Assertions.assertEquals(0f, result.matrix[1, 1])
    }

    @Test
    fun `RREF is idempotent`() {
        val matrix = Matrix(
            floatArrayOf(2f, 4f, 6f),
            floatArrayOf(1f, 3f, 5f),
        )

        val first = matrix.rref()
        val second = first.matrix.rref()

        Assertions.assertEquals(first.pivotColumns, second.pivotColumns)

        for (row in 0 until matrix.rows) {
            for (column in 0 until matrix.columns) {
                Assertions.assertEquals(
                    first.matrix[row, column],
                    second.matrix[row, column],
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `RREF rank agrees with matrix rank`() {
        val matrices = listOf(
            Matrix.identity(3),
            Matrix.diagonal(1f, 0f, 3f),
            Matrix(
                floatArrayOf(1f, 2f, 3f),
                floatArrayOf(2f, 4f, 6f),
            ),
            Matrix(2, 4) { _, _ -> 0f },
        )

        for (matrix in matrices) {
            Assertions.assertEquals(matrix.rank(), matrix.rref().rank)
        }
    }

    @Test
    fun `RREF is invariant under nonzero uniform scaling`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(0f, 1f, 4f),
        )

        val original = matrix.rref()
        val scaled = (matrix * 1e-8f).rref()

        Assertions.assertEquals(original.pivotColumns, scaled.pivotColumns)

        for (row in 0 until matrix.rows) {
            for (column in 0 until matrix.columns) {
                Assertions.assertEquals(
                    original.matrix[row, column],
                    scaled.matrix[row, column],
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `tolerance controls numerical rank`() {
        val matrix = Matrix.diagonal(1f, 1e-5f)

        Assertions.assertEquals(2, matrix.rref(1e-6f).rank)
        Assertions.assertEquals(1, matrix.rref(1e-4f).rank)
    }

    @Test
    fun `RREF rejects invalid tolerance`() {
        val matrix = Matrix.identity(2)

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            matrix.rref(-1f)
        }

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            matrix.rref(Float.NaN)
        }
    }

    @Test
    fun `RREF rejects non finite matrices`() {
        val matrix = Matrix(
            floatArrayOf(1f, Float.POSITIVE_INFINITY),
            floatArrayOf(0f, 1f),
        )

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            matrix.rref()
        }
    }
}