
package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import org.soyuz.vektors.tensor.All
import org.soyuz.vektors.tensor.MutableTensor
import org.soyuz.vektors.tensor.Slice
import org.soyuz.vektors.tensor.Tensor
import org.soyuz.vektors.vector.Vector

class TensorContractionTest {

    @Test
    fun `vector contraction computes dot product`() {
        val a = Vector(1f, 2f, 3f).toTensor()
        val b = Vector(4f, 5f, 6f).toTensor()

        val result = a.contract(
            b,
            intArrayOf(0),
            intArrayOf(0),
        )

        assertEquals(emptyList<Int>(), result.shape)
        assertEquals(32f, result.valueAt())
    }

    @Test
    fun `contraction computes matrix multiplication`() {
        val a = Tensor(2, 3) { (i, k) ->
            (i * 3 + k + 1).toFloat()
        }

        val b = Tensor(3, 2) { (k, j) ->
            (k * 2 + j + 1).toFloat()
        }

        val result = a.contract(
            b,
            intArrayOf(1),
            intArrayOf(0),
        )

        assertEquals(listOf(2, 2), result.shape)

        assertEquals(22f, result.valueAt(0, 0))
        assertEquals(28f, result.valueAt(0, 1))
        assertEquals(49f, result.valueAt(1, 0))
        assertEquals(64f, result.valueAt(1, 1))
    }

    @Test
    fun `contraction computes matrix vector multiplication`() {
        val a = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val b = Tensor(3) { (i) ->
            (i + 1).toFloat()
        }

        val result = a.contract(
            b,
            intArrayOf(1),
            intArrayOf(0),
        )

        assertEquals(listOf(2), result.shape)
        assertEquals(14f, result.valueAt(0))
        assertEquals(32f, result.valueAt(1))
    }

    @Test
    fun `contraction supports multiple axes`() {
        val a = Tensor(2, 3, 4) { (i, j, k) ->
            (i + j + k).toFloat()
        }

        val b = Tensor(3, 4, 5) { (j, k, l) ->
            (j + k + l).toFloat()
        }

        val result = a.contract(
            b,
            intArrayOf(1, 2),
            intArrayOf(0, 1),
        )

        assertEquals(listOf(2, 5), result.shape)

        for (i in 0 until 2) {
            for (l in 0 until 5) {
                var expected = 0f

                for (j in 0 until 3) {
                    for (k in 0 until 4) {
                        expected +=
                            a.valueAt(i, j, k) *
                                    b.valueAt(j, k, l)
                    }
                }

                assertEquals(
                    expected,
                    result.valueAt(i, l),
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `contraction pairs axes in specified order`() {
        val a = Tensor(2, 3) { (i, j) ->
            (i * 10 + j).toFloat()
        }

        val b = Tensor(3, 2) { (j, i) ->
            (j * 10 + i).toFloat()
        }

        val result = a.contract(
            b,
            intArrayOf(1, 0),
            intArrayOf(0, 1),
        )

        var expected = 0f

        for (i in 0 until 2) {
            for (j in 0 until 3) {
                expected += a.valueAt(i, j) * b.valueAt(j, i)
            }
        }

        assertEquals(emptyList<Int>(), result.shape)
        assertEquals(expected, result.valueAt())
    }

    @Test
    fun `empty contraction axes produce outer product`() {
        val a = Tensor(2) { (i) ->
            (i + 1).toFloat()
        }

        val b = Tensor(3) { (i) ->
            (i + 2).toFloat()
        }

        val result = a.contract(
            b,
            intArrayOf(),
            intArrayOf(),
        )

        assertEquals(listOf(2, 3), result.shape)

        assertEquals(2f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(4f, result.valueAt(0, 2))
        assertEquals(4f, result.valueAt(1, 0))
        assertEquals(6f, result.valueAt(1, 1))
        assertEquals(8f, result.valueAt(1, 2))
    }

    @Test
    fun `scalar contraction multiplies values`() {
        val a = Tensor.scalar(3f)
        val b = Tensor.scalar(4f)

        val result = a.contract(
            b,
            intArrayOf(),
            intArrayOf(),
        )

        assertEquals(emptyList<Int>(), result.shape)
        assertEquals(12f, result.valueAt())
    }

    @Test
    fun `contraction supports sliced tensor views`() {
        val source = Tensor(6, 4) { (i, j) ->
            (i * 10 + j).toFloat()
        }

        val view = source[
            Slice(0, 6, 2),
            All,
        ]

        val b = Tensor(4, 2) { (i, j) ->
            (i + j + 1).toFloat()
        }

        val result = view.contract(
            b,
            intArrayOf(1),
            intArrayOf(0),
        )

        assertEquals(listOf(3, 2), result.shape)

        for (i in 0 until 3) {
            for (j in 0 until 2) {
                var expected = 0f

                for (k in 0 until 4) {
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
    fun `contraction supports permuted tensor views`() {
        val source = Tensor(2, 3, 4) { (i, j, k) ->
            (i * 100 + j * 10 + k).toFloat()
        }

        val permuted = source.T[2, 0, 1]

        val b = Tensor(3) { (i) ->
            (i + 1).toFloat()
        }

        val result = permuted.contract(
            b,
            intArrayOf(2),
            intArrayOf(0),
        )

        assertEquals(listOf(4, 2), result.shape)

        for (k in 0 until 4) {
            for (i in 0 until 2) {
                var expected = 0f

                for (j in 0 until 3) {
                    expected +=
                        source.valueAt(i, j, k) *
                                b.valueAt(j)
                }

                assertEquals(
                    expected,
                    result.valueAt(k, i),
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `contraction supports mutable tensors`() {
        val a = MutableTensor(2, 2) { (i, j) ->
            (i + j + 1).toFloat()
        }

        val b = Tensor(2) { (i) ->
            (i + 1).toFloat()
        }

        val result = a.contract(
            b,
            intArrayOf(1),
            intArrayOf(0),
        )

        assertEquals(5f, result.valueAt(0))
        assertEquals(8f, result.valueAt(1))

        a.setValueAt(0, 0, value = 100f)

        assertEquals(5f, result.valueAt(0))
    }

    @Test
    fun `contraction rejects mismatched dimensions`() {
        val a = Tensor(2, 3) { 1f }
        val b = Tensor(4, 2) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            a.contract(
                b,
                intArrayOf(1),
                intArrayOf(0),
            )
        }
    }

    @Test
    fun `contraction rejects mismatched axis counts`() {
        val a = Tensor(2, 3) { 1f }
        val b = Tensor(3, 2) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            a.contract(
                b,
                intArrayOf(0, 1),
                intArrayOf(0),
            )
        }
    }

    @Test
    fun `contraction rejects duplicate axes`() {
        val a = Tensor(2, 3) { 1f }
        val b = Tensor(2, 3) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            a.contract(
                b,
                intArrayOf(0, 0),
                intArrayOf(0, 1),
            )
        }
    }

    @Test
    fun `contraction rejects invalid axes`() {
        val a = Tensor(2, 3) { 1f }
        val b = Tensor(3, 2) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            a.contract(
                b,
                intArrayOf(2),
                intArrayOf(0),
            )
        }
    }
}
