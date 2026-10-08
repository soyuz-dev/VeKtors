package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class TensorIndexingTest {
    @Test
    fun `single indexing removes the first dimension`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val row = tensor[1]
        val scalar = row[2]

        assertEquals(listOf(3), row.shape)
        assertEquals(emptyList<Int>(), scalar.shape)

        assertEquals(12f, row.valueAt(2))
        assertEquals(12f, scalar.valueAt())
    }

    @Test
    fun `chained indexing matches valueAt`() {
        val tensor = Tensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        assertEquals(
            tensor.valueAt(1, 2, 3),
            tensor[1][2][3].valueAt(),
        )
    }

    @Test
    fun `slice creates correct immutable view`() {
        val tensor = Tensor(6, 4) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val view = tensor[
            Slice(1, 6, 2),
            Slice(1, 4),
        ]

        assertEquals(listOf(3, 3), view.shape)

        assertEquals(11f, view.valueAt(0, 0))
        assertEquals(13f, view.valueAt(0, 2))

        assertEquals(31f, view.valueAt(1, 0))
        assertEquals(33f, view.valueAt(1, 2))

        assertEquals(51f, view.valueAt(2, 0))
        assertEquals(53f, view.valueAt(2, 2))
    }

    @Test
    fun `index removes dimension`() {
        val tensor = Tensor(6, 4) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val view = tensor[
            Index(3),
            All,
        ]

        assertEquals(listOf(4), view.shape)

        assertEquals(30f, view.valueAt(0))
        assertEquals(31f, view.valueAt(1))
        assertEquals(32f, view.valueAt(2))
        assertEquals(33f, view.valueAt(3))
    }

    @Test
    fun `range preserves dimension`() {
        val tensor = Tensor(6, 4) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val view = tensor[
            Range(1..3),
            All,
        ]

        assertEquals(listOf(3, 4), view.shape)

        assertEquals(10f, view.valueAt(0, 0))
        assertEquals(13f, view.valueAt(0, 3))

        assertEquals(20f, view.valueAt(1, 0))
        assertEquals(23f, view.valueAt(1, 3))

        assertEquals(30f, view.valueAt(2, 0))
        assertEquals(33f, view.valueAt(2, 3))
    }

    @Test
    fun `all preserves complete dimension`() {
        val tensor = Tensor(3, 4) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val view = tensor[
            All,
            Index(2),
        ]

        assertEquals(listOf(3), view.shape)

        assertEquals(2f, view.valueAt(0))
        assertEquals(12f, view.valueAt(1))
        assertEquals(22f, view.valueAt(2))
    }

    @Test
    fun `missing tensor indices imply all`() {
        val tensor = Tensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val implicit = tensor[
            Index(1),
        ]

        val explicit = tensor[
            Index(1),
            All,
            All,
        ]

        assertEquals(listOf(3, 4), implicit.shape)
        assertEquals(explicit.shape, implicit.shape)

        for (y in 0 until 3) {
            for (z in 0 until 4) {
                assertEquals(
                    explicit.valueAt(y, z),
                    implicit.valueAt(y, z),
                )
            }
        }
    }

    @Test
    fun `mixed advanced indexing works`() {
        val tensor = Tensor(4, 5, 6) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val result = tensor[
            Index(2),
            Range(1..3),
            Slice(0, 6, 2),
        ]

        assertEquals(listOf(3, 3), result.shape)

        assertEquals(210f, result.valueAt(0, 0))
        assertEquals(212f, result.valueAt(0, 1))
        assertEquals(214f, result.valueAt(0, 2))

        assertEquals(220f, result.valueAt(1, 0))
        assertEquals(222f, result.valueAt(1, 1))
        assertEquals(224f, result.valueAt(1, 2))

        assertEquals(230f, result.valueAt(2, 0))
        assertEquals(232f, result.valueAt(2, 1))
        assertEquals(234f, result.valueAt(2, 2))
    }

    @Test
    fun `indexing all dimensions produces scalar tensor`() {
        val tensor = Tensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val scalar = tensor[
            Index(1),
            Index(2),
            Index(3),
        ]

        assertEquals(emptyList<Int>(), scalar.shape)
        assertEquals(0, scalar.rank)
        assertEquals(1, scalar.size)
        assertEquals(123f, scalar.valueAt())
    }

    @Test
    fun `range and slice have different end semantics`() {
        val tensor = Tensor(10) { (x) ->
            x.toFloat()
        }

        val range = tensor[
            Range(2..7),
        ]

        val slice = tensor[
            Slice(2, 7),
        ]

        assertEquals(listOf(6), range.shape)
        assertEquals(listOf(5), slice.shape)

        assertEquals(7f, range.valueAt(5))
        assertEquals(6f, slice.valueAt(4))
    }

    @Test
    fun `stepped slice uses correct stride`() {
        val tensor = Tensor(10) { (x) ->
            x.toFloat()
        }

        val view = tensor[
            Slice(1, 10, 3),
        ]

        assertEquals(listOf(3), view.shape)

        assertEquals(1f, view.valueAt(0))
        assertEquals(4f, view.valueAt(1))
        assertEquals(7f, view.valueAt(2))
    }

    @Test
    fun `invalid index throws`() {
        val tensor = Tensor(2, 3) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor[Index(2)]
        }
    }

    @Test
    fun `invalid range throws`() {
        val tensor = Tensor(2, 3) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor[
                All,
                Range(1..3),
            ]
        }
    }

    @Test
    fun `too many indices throws`() {
        val tensor = Tensor(2, 3) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor[
                All,
                All,
                All,
            ]
        }
    }
}
