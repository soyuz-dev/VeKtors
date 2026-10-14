
package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class TensorMatmulTest {

    @Test
    fun `matmul computes ordinary matrix multiplication`() {
        val a = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val b = Tensor(3, 2) { (i, j) ->
            (i * 2 + j + 1).toFloat()
        }

        val result = a matmul b

        assertEquals(listOf(2, 2), result.shape)
        assertEquals(22f, result.valueAt(0, 0))
        assertEquals(28f, result.valueAt(0, 1))
        assertEquals(49f, result.valueAt(1, 0))
        assertEquals(64f, result.valueAt(1, 1))
    }

    @Test
    fun `matmul matches ordinary Matrix multiplication`() {
        val a = Matrix(2, 3) { i, j ->
            (i + j + 1).toFloat()
        }
        val b = Matrix(3, 4) { i, j ->
            (i * 2 + j + 1).toFloat()
        }

        val expected = a * b
        val actual = a.toTensor() matmul b.toTensor()

        for (i in 0 until expected.rows) {
            for (j in 0 until expected.columns) {
                assertEquals(
                    expected[i, j],
                    actual.valueAt(i, j),
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `matmul supports batched matrices`() {
        val a = Tensor(2, 2, 3) { (batch, i, k) ->
            (batch + i + k + 1).toFloat()
        }

        val b = Tensor(2, 3, 2) { (batch, k, j) ->
            (batch + k + j + 1).toFloat()
        }

        val result = a matmul b

        assertEquals(listOf(2, 2, 2), result.shape)

        for (batch in 0 until 2) {
            for (i in 0 until 2) {
                for (j in 0 until 2) {
                    var expected = 0f

                    for (k in 0 until 3) {
                        expected +=
                            a.valueAt(batch, i, k) *
                                    b.valueAt(batch, k, j)
                    }

                    assertEquals(
                        expected,
                        result.valueAt(batch, i, j),
                        1e-5f,
                    )
                }
            }
        }
    }

    @Test
    fun `matmul broadcasts singleton batch dimension`() {
        val a = Tensor(1, 2, 3) { (_, i, k) ->
            (i + k + 1).toFloat()
        }

        val b = Tensor(4, 3, 2) { (batch, k, j) ->
            (batch + k + j + 1).toFloat()
        }

        val result = a matmul b

        assertEquals(listOf(4, 2, 2), result.shape)

        for (batch in 0 until 4) {
            for (i in 0 until 2) {
                for (j in 0 until 2) {
                    var expected = 0f

                    for (k in 0 until 3) {
                        expected +=
                            a.valueAt(0, i, k) *
                                    b.valueAt(batch, k, j)
                    }

                    assertEquals(
                        expected,
                        result.valueAt(batch, i, j),
                        1e-5f,
                    )
                }
            }
        }
    }

    @Test
    fun `matmul supports different batch ranks`() {
        val a = Tensor(2, 1, 3, 4) { (x, _, i, k) ->
            (x + i + k + 1).toFloat()
        }

        val b = Tensor(1, 5, 4, 2) { (_, y, k, j) ->
            (y + k + j + 1).toFloat()
        }

        val result = a matmul b

        assertEquals(listOf(2, 5, 3, 2), result.shape)

        for (x in 0 until 2) {
            for (y in 0 until 5) {
                for (i in 0 until 3) {
                    for (j in 0 until 2) {
                        var expected = 0f

                        for (k in 0 until 4) {
                            expected +=
                                a.valueAt(x, 0, i, k) *
                                        b.valueAt(0, y, k, j)
                        }

                        assertEquals(
                            expected,
                            result.valueAt(x, y, i, j),
                            1e-5f,
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `matmul broadcasts unbatched matrix`() {
        val a = Tensor(2, 3) { (i, k) ->
            (i + k + 1).toFloat()
        }

        val b = Tensor(4, 3, 2) { (batch, k, j) ->
            (batch + k + j + 1).toFloat()
        }

        val result = a matmul b

        assertEquals(listOf(4, 2, 2), result.shape)
    }

    @Test
    fun `matmul supports sliced views`() {
        val source = Tensor(6, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val a = source[Slice(0, 6, 2), All]
        val b = Tensor(3, 2) { (i, j) ->
            (i + j + 1).toFloat()
        }

        val result = a matmul b

        assertEquals(listOf(3, 2), result.shape)

        for (i in 0 until 3) {
            for (j in 0 until 2) {
                var expected = 0f

                for (k in 0 until 3) {
                    expected +=
                        source.valueAt(i * 2, k) *
                                b.valueAt(k, j)
                }

                assertEquals(
                    expected,
                    result.valueAt(i, j),
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `matmul supports permuted views`() {
        val source = Tensor(2, 3, 4) { (i, j, k) ->
            (i * 100 + j * 10 + k).toFloat()
        }

        val a = source.T[0, 2, 1]
        val b = Tensor(3, 2) { (i, j) ->
            (i + j + 1).toFloat()
        }

        val result = a matmul b

        assertEquals(listOf(2, 4, 2), result.shape)

        for (batch in 0 until 2) {
            for (i in 0 until 4) {
                for (j in 0 until 2) {
                    var expected = 0f

                    for (k in 0 until 3) {
                        expected +=
                            source.valueAt(batch, k, i) *
                                    b.valueAt(k, j)
                    }

                    assertEquals(
                        expected,
                        result.valueAt(batch, i, j),
                        1e-5f,
                    )
                }
            }
        }
    }

    @Test
    fun `matmul supports mutable inputs without sharing output`() {
        val a = MutableTensor(2, 2) { (i, j) ->
            (i * 2 + j + 1).toFloat()
        }

        val b = Tensor(2, 2) { (i, j) ->
            if (i == j) 1f else 0f
        }

        val result = a matmul b

        a.setValueAt(0, 0, value = 100f)

        assertEquals(1f, result.valueAt(0, 0))
        assertEquals(100f, a.valueAt(0, 0))
    }

    @Test
    fun `matmul rejects incompatible inner dimensions`() {
        val a = Tensor(2, 3) { 1f }
        val b = Tensor(4, 2) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            a matmul b
        }
    }

    @Test
    fun `matmul rejects incompatible batch dimensions`() {
        val a = Tensor(2, 3, 4) { 1f }
        val b = Tensor(5, 4, 6) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            a matmul b
        }
    }

    @Test
    fun `matmul rejects scalar and vector operands`() {
        val matrix = Tensor(2, 2) { 1f }
        val vector = Tensor(2) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            matrix matmul vector
        }

        assertThrows(IllegalArgumentException::class.java) {
            Tensor.scalar(1f) matmul matrix
        }
    }

    @Test
    fun `outer produces expected matrix`() {
        val a = Tensor(2) { (i) ->
            (i + 1).toFloat()
        }

        val b = Tensor(3) { (i) ->
            (i + 2).toFloat()
        }

        val result = a outer b

        assertEquals(listOf(2, 3), result.shape)
        assertEquals(2f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(4f, result.valueAt(0, 2))
        assertEquals(4f, result.valueAt(1, 0))
        assertEquals(6f, result.valueAt(1, 1))
        assertEquals(8f, result.valueAt(1, 2))
    }

    @Test
    fun `outer works with scalar tensors`() {
        val scalar = Tensor.scalar(3f)
        val vector = Tensor(2) { (i) ->
            (i + 1).toFloat()
        }

        val result = scalar outer vector

        assertEquals(listOf(2), result.shape)
        assertEquals(3f, result.valueAt(0))
        assertEquals(6f, result.valueAt(1))
    }
}
