package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class MatrixSolveTest {
    @Test
    fun `solve two by two system`() {
        val a = Matrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(1f, 3f),
        )

        val b = Vector(5f, 7f)
        val x = a.solve(b)

        assertEquals(1.6f, x[0], 0.00001f)
        assertEquals(1.8f, x[1], 0.00001f)
    }

    @Test
    fun `solve identity system`() {
        val a = Matrix.identity(3)
        val b = Vector(4f, 5f, 6f)

        val x = a.solve(b)

        for (i in 0 until 3) {
            assertEquals(b[i], x[i], 0.00001f)
        }
    }

    @Test
    fun `solve diagonal system`() {
        val a = Matrix.diagonal(2f, 4f, 5f)
        val b = Vector(6f, 12f, 20f)

        val x = a.solve(b)

        assertEquals(3f, x[0], 0.00001f)
        assertEquals(3f, x[1], 0.00001f)
        assertEquals(4f, x[2], 0.00001f)
    }

    @Test
    fun `solve system requiring pivoting`() {
        val a = Matrix(
            floatArrayOf(0f, 2f),
            floatArrayOf(3f, 4f),
        )

        val b = Vector(8f, 18f)
        val x = a.solve(b)

        assertEquals(2f / 3f, x[0], 0.00001f)
        assertEquals(4f, x[1], 0.00001f)
    }

    @Test
    fun `solve three by three system`() {
        val a = Matrix(
            floatArrayOf(3f, 2f, -1f),
            floatArrayOf(2f, -2f, 4f),
            floatArrayOf(-1f, 0.5f, -1f),
        )

        val b = Vector(1f, -2f, 0f)
        val x = a.solve(b)

        assertEquals(1f, x[0], 0.00001f)
        assertEquals(-2f, x[1], 0.00001f)
        assertEquals(-2f, x[2], 0.00001f)
    }

    @Test
    fun `solve rejects singular matrix`() {
        val a = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(2f, 4f),
        )

        assertThrows(IllegalArgumentException::class.java) {
            a.solve(Vector(3f, 6f))
        }
    }

    @Test
    fun `solve rejects mismatched vector size`() {
        val a = Matrix.identity(3)

        assertThrows(IllegalArgumentException::class.java) {
            a.solve(Vector(1f, 2f))
        }
    }

    @Test
    fun `solve does not mutate inputs`() {
        val a = MutableMatrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(1f, 3f),
        )
        val b = MutableVector(5f, 7f)

        val x = a.solve(b)

        assertEquals(1.6f, x[0], 0.00001f)
        assertEquals(1.8f, x[1], 0.00001f)

        assertEquals(2f, a[0, 0])
        assertEquals(1f, a[0, 1])
        assertEquals(1f, a[1, 0])
        assertEquals(3f, a[1, 1])

        assertEquals(5f, b[0])
        assertEquals(7f, b[1])
    }
}
