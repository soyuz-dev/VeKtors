package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class MatrixTest {
    @Test
    fun `immutable identity matrix works`() {
        val matrix = Matrix.identity(4)

        assertEquals(4, matrix.rows)
        assertEquals(4, matrix.columns)

        for (row in 0 until 4) {
            for (column in 0 until 4) {
                assertEquals(
                    if (row == column) 1f else 0f,
                    matrix[row, column],
                )
            }
        }
    }

    @Test
    fun `mutable identity matrix can be modified`() {
        val matrix = MutableMatrix.identity(3)

        assertEquals(1f, matrix[1, 1])

        matrix[1, 1] = 5f

        assertEquals(5f, matrix[1, 1])
        assertEquals(1f, matrix[0, 0])
    }

    @Test
    fun `immutable diagonal matrix works`() {
        val matrix = Matrix.diagonal(2f, 4f, 8f)

        assertEquals(3, matrix.rows)
        assertEquals(3, matrix.columns)

        for (row in 0 until 3) {
            for (column in 0 until 3) {
                val expected = if (row == column) {
                    floatArrayOf(2f, 4f, 8f)[row]
                } else {
                    0f
                }

                assertEquals(expected, matrix[row, column])
            }
        }
    }

    @Test
    fun `mutable diagonal matrix can be modified`() {
        val matrix = MutableMatrix.diagonal(2f, 4f, 8f)

        assertEquals(4f, matrix[1, 1])

        matrix[1, 1] = 10f

        assertEquals(10f, matrix[1, 1])
        assertEquals(2f, matrix[0, 0])
        assertEquals(8f, matrix[2, 2])
    }

    @Test
    fun `matrix trace works for both mutable and immutable matrices`() {
        val immutable = Matrix.diagonal(2f, 4f, 8f)
        val mutable = MutableMatrix.diagonal(2f, 4f, 8f)

        assertEquals(14f, immutable.trace())
        assertEquals(14f, mutable.trace())
    }

    @Test
    fun `matrix trace rejects non square matrices`() {
        val matrix = Matrix(2, 3) { _, _ -> 0f }

        assertThrows(IllegalArgumentException::class.java) {
            matrix.trace()
        }
    }

    @Test
    fun `determinant of identity is one`() {
        val matrix = Matrix.identity(4)

        assertEquals(1f, matrix.determinant())
    }

    @Test
    fun `determinant of diagonal matrix is product of diagonal`() {
        val matrix = Matrix.diagonal(2f, 3f, 4f)

        assertEquals(24f, matrix.determinant())
    }

    @Test
    fun `determinant of general matrix works`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(0f, 1f, 4f),
            floatArrayOf(5f, 6f, 0f),
        )

        assertEquals(1f, matrix.determinant(), 1e-5f)
    }

    @Test
    fun `determinant handles row swapping`() {
        val matrix = Matrix(
            floatArrayOf(0f, 1f),
            floatArrayOf(2f, 3f),
        )

        assertEquals(-2f, matrix.determinant())
    }

    @Test
    fun `singular matrix has zero determinant`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(2f, 4f),
        )

        assertEquals(0f, matrix.determinant())
    }

    @Test
    fun `inverse of identity is identity`() {
        val inverse = Matrix.identity(3).inverse()

        for (row in 0 until 3) {
            for (column in 0 until 3) {
                assertEquals(
                    if (row == column) 1f else 0f,
                    inverse[row, column],
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `inverse of two by two matrix works`() {
        val matrix = Matrix(
            floatArrayOf(4f, 7f),
            floatArrayOf(2f, 6f),
        )

        val inverse = matrix.inverse()

        assertEquals(0.6f, inverse[0, 0], 1e-5f)
        assertEquals(-0.7f, inverse[0, 1], 1e-5f)
        assertEquals(-0.2f, inverse[1, 0], 1e-5f)
        assertEquals(0.4f, inverse[1, 1], 1e-5f)
    }

    @Test
    fun `matrix multiplied by inverse produces identity`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(0f, 1f, 4f),
            floatArrayOf(5f, 6f, 0f),
        )

        val result = matrix * matrix.inverse()

        for (row in 0 until 3) {
            for (column in 0 until 3) {
                assertEquals(
                    if (row == column) 1f else 0f,
                    result[row, column],
                    1e-4f,
                )
            }
        }
    }

    @Test
    fun `mutable matrix supports determinant and inverse`() {
        val matrix = MutableMatrix(
            floatArrayOf(4f, 7f),
            floatArrayOf(2f, 6f),
        )

        assertEquals(10f, matrix.determinant(), 1e-5f)

        val inverse = matrix.inverse()

        assertEquals(0.6f, inverse[0, 0], 1e-5f)
        assertEquals(4f, matrix[0, 0])
        assertEquals(7f, matrix[0, 1])
    }

    @Test
    fun `singular matrix cannot be inverted`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(2f, 4f),
        )

        assertThrows(IllegalArgumentException::class.java) {
            matrix.inverse()
        }
    }

    @Test
    fun `non square matrices reject determinant and inverse`() {
        val matrix = Matrix(2, 3) { row, column ->
            (row * 10 + column).toFloat()
        }

        assertThrows(IllegalArgumentException::class.java) {
            matrix.determinant()
        }

        assertThrows(IllegalArgumentException::class.java) {
            matrix.inverse()
        }
    }
}
