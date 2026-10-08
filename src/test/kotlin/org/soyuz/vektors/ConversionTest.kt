package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class ConversionTest {
    @Test
    fun `vector converts to tensor`() {
        val vector = Vector(1f, 2f, 3f)
        val tensor = vector.toTensor()

        assertEquals(listOf(3), tensor.shape)
        assertEquals(1f, tensor.valueAt(0))
        assertEquals(2f, tensor.valueAt(1))
        assertEquals(3f, tensor.valueAt(2))
    }

    @Test
    fun `matrix converts to tensor`() {
        val matrix = Matrix(2, 3) { row, column ->
            (row * 10 + column).toFloat()
        }

        val tensor = matrix.toTensor()

        assertEquals(listOf(2, 3), tensor.shape)
        assertEquals(12f, tensor.valueAt(1, 2))
    }

    @Test
    fun `rank one tensor converts to vector`() {
        val tensor = Tensor(3) { (i) ->
            (i + 1).toFloat()
        }

        val vector = tensor.toVector()

        assertEquals(3, vector.size)
        assertEquals(3f, vector[2])
    }

    @Test
    fun `rank two tensor converts to matrix`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val matrix = tensor.toMatrix()

        assertEquals(2, matrix.rows)
        assertEquals(3, matrix.columns)
        assertEquals(12f, matrix[1, 2])
    }

    @Test
    fun `higher rank tensor cannot convert to matrix`() {
        val tensor = Tensor(2, 3, 4) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.toMatrix()
        }
    }

    @Test
    fun `vector round trip preserves values`() {
        val vector = Vector(1f, 2f, 3f)

        val result =
            vector
                .toTensor()
                .toVector()

        assertEquals(vector.size, result.size)

        for (i in 0 until vector.size) {
            assertEquals(vector[i], result[i])
        }
    }

    @Test
    fun `matrix round trip preserves values`() {
        val matrix = Matrix(2, 3) { row, column ->
            (row * 10 + column).toFloat()
        }

        val result =
            matrix
                .toTensor()
                .toMatrix()

        assertEquals(matrix.rows, result.rows)
        assertEquals(matrix.columns, result.columns)

        for (row in 0 until matrix.rows) {
            for (column in 0 until matrix.columns) {
                assertEquals(
                    matrix[row, column],
                    result[row, column],
                )
            }
        }
    }

    @Test
    fun `rank two tensor cannot convert to vector`() {
        val tensor = Tensor(2, 3) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.toVector()
        }
    }

    @Test
    fun `rank three tensor cannot convert to matrix`() {
        val tensor = Tensor(2, 3, 4) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.toMatrix()
        }
    }
}
