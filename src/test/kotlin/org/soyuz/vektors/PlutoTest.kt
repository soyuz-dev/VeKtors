package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class PlutoTest {
    @Test
    fun `LU determinant`() {
        val a = Matrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(3f, 4f),
        )

        assertEquals(5f, a.lu.determinant, 1e-5f)
        assertEquals(5f, a.determinant(), 1e-5f)
    }

    @Test
    fun `LU pivoting`() {
        val a = Matrix(
            floatArrayOf(0f, 2f),
            floatArrayOf(3f, 4f),
        )

        val lu = a.lu

        assertEquals(-6f, lu.determinant, 1e-5f)

        val x = lu.solve(Vector(8f, 18f))

        assertEquals(2f / 3f, x[0], 1e-5f)
        assertEquals(4f, x[1], 1e-5f)
    }

    @Test
    fun `LU solve multiple right hand sides`() {
        val a = Matrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(1f, 3f),
        )

        val b = Matrix(
            floatArrayOf(5f, 1f),
            floatArrayOf(7f, 4f),
        )

        val x = a.lu.solve(b)
        val reconstructed = a * x

        for (row in 0 until b.rows) {
            for (column in 0 until b.columns) {
                assertEquals(
                    b[row, column],
                    reconstructed[row, column],
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `LU inverse`() {
        val a = Matrix(
            floatArrayOf(4f, 7f),
            floatArrayOf(2f, 6f),
        )

        val result = a * a.lu.inverse

        for (row in 0 until result.rows) {
            for (column in 0 until result.columns) {
                assertEquals(
                    if (row == column) 1f else 0f,
                    result[row, column],
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `LU singular matrix`() {
        val a = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(2f, 4f),
        )

        val lu = a.lu

        assertTrue(lu.isSingular)
        assertEquals(0f, lu.determinant)

        assertThrows(IllegalArgumentException::class.java) {
            lu.solve(Vector(3f, 6f))
        }

        assertThrows(IllegalArgumentException::class.java) {
            lu.inverse
        }
    }

    @Test
    fun `LU rejects non square matrix`() {
        val a = Matrix(2, 3) { _, _ -> 1f }

        assertThrows(IllegalArgumentException::class.java) {
            a.lu
        }
    }

    @Test
    fun `LU is independent of mutable matrix`() {
        val a = MutableMatrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(1f, 3f),
        )

        val lu = a.lu

        a[0, 0] = 100f

        assertEquals(5f, lu.determinant, 1e-5f)
        assertEquals(299f, a.determinant(), 1e-4f)
    }

    @Test
    fun `LU can be reused for separate systems`() {
        val a = Matrix(
            floatArrayOf(3f, 1f),
            floatArrayOf(1f, 2f),
        )

        val lu = a.lu

        val x1 = lu.solve(Vector(7f, 5f))
        val x2 = lu.solve(Vector(4f, 3f))

        assertEquals(1.8f, x1[0], 1e-5f)
        assertEquals(1.6f, x1[1], 1e-5f)

        assertEquals(1f, x2[0], 1e-5f)
        assertEquals(1f, x2[1], 1e-5f)
    }
}
