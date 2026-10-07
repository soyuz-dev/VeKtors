package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import kotlin.math.sqrt

class TensorTest {

    @Test
    fun `basic tensor properties work`() {
        val tensor = Tensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        assertEquals(listOf(2, 3, 4), tensor.shape)
        assertEquals(3, tensor.rank)
        assertEquals(24, tensor.size)

        assertEquals(0f, tensor.valueAt(0, 0, 0))
        assertEquals(123f, tensor.valueAt(1, 2, 3))
    }

    @Test
    fun `scalar tensor works`() {
        val scalar = Tensor.scalar(42f)

        assertEquals(emptyList<Int>(), scalar.shape)
        assertEquals(0, scalar.rank)
        assertEquals(1, scalar.size)
        assertEquals(42f, scalar.valueAt())
        assertEquals(42f, scalar.sum())
    }

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
    fun `sum mean min and max work`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        assertEquals(36f, tensor.sum())
        assertEquals(6f, tensor.mean())
        assertEquals(0f, tensor.min())
        assertEquals(12f, tensor.max())
    }

    @Test
    fun `map creates transformed tensor`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val doubled = tensor.map { it * 2f }

        assertEquals(0f, doubled.valueAt(0, 0))
        assertEquals(4f, doubled.valueAt(0, 2))
        assertEquals(24f, doubled.valueAt(1, 2))
    }

    @Test
    fun `reshape preserves flattened values`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 3 + column).toFloat()
        }

        val reshaped = tensor.reshape(3, 2)

        assertEquals(listOf(3, 2), reshaped.shape)

        assertEquals(0f, reshaped.valueAt(0, 0))
        assertEquals(1f, reshaped.valueAt(0, 1))
        assertEquals(2f, reshaped.valueAt(1, 0))
        assertEquals(3f, reshaped.valueAt(1, 1))
        assertEquals(4f, reshaped.valueAt(2, 0))
        assertEquals(5f, reshaped.valueAt(2, 1))
    }

    @Test
    fun `invalid reshape throws`() {
        val tensor = Tensor(2, 3) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.reshape(4, 2)
        }
    }

    @Test
    fun `elementwise addition works`() {
        val a = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val b = Tensor(2, 3) { (_, column) ->
            (column + 1).toFloat()
        }

        val result = a + b

        assertEquals(1f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(5f, result.valueAt(0, 2))

        assertEquals(11f, result.valueAt(1, 0))
        assertEquals(13f, result.valueAt(1, 1))
        assertEquals(15f, result.valueAt(1, 2))
    }

    @Test
    fun `broadcasting works`() {
        val a = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val b = Tensor(3) { (column) ->
            (column + 1).toFloat()
        }

        val result = a + b

        assertEquals(listOf(2, 3), result.shape)

        assertEquals(1f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(5f, result.valueAt(0, 2))

        assertEquals(11f, result.valueAt(1, 0))
        assertEquals(13f, result.valueAt(1, 1))
        assertEquals(15f, result.valueAt(1, 2))
    }

    @Test
    fun `scalar multiplication works`() {
        val tensor = Tensor(2, 2) { (row, column) ->
            (row * 2 + column).toFloat()
        }

        val result = tensor * 3f

        assertEquals(0f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(6f, result.valueAt(1, 0))
        assertEquals(9f, result.valueAt(1, 1))
    }

    @Test
    fun `mutable tensor can change values`() {
        val tensor = MutableTensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        assertEquals(12f, tensor.valueAt(1, 2))

        tensor.setValueAt(
            1,
            2,
            value = 69f,
        )

        assertEquals(69f, tensor.valueAt(1, 2))
    }

    @Test
    fun `mutable indexed views share backing storage`() {
        val tensor = MutableTensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val row = tensor[1]

        assertEquals(listOf(3), row.shape)
        assertEquals(12f, row.valueAt(2))

        row *= 2f

        assertEquals(24f, row.valueAt(2))
        assertEquals(24f, tensor.valueAt(1, 2))
    }

    @Test
    fun `mutable scalar views share backing storage`() {
        val tensor = MutableTensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val scalar = tensor[1][2]

        assertEquals(emptyList<Int>(), scalar.shape)
        assertEquals(12f, scalar.valueAt())

        scalar *= 10f

        assertEquals(120f, scalar.valueAt())
        assertEquals(120f, tensor.valueAt(1, 2))
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
    fun `mutable slice view shares backing storage`() {
        val tensor = MutableTensor(6, 4) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val view = tensor[
            Slice(1, 6, 2),
            Slice(1, 4),
        ]

        assertEquals(listOf(3, 3), view.shape)

        view *= 10f

        assertEquals(110f, tensor.valueAt(1, 1))
        assertEquals(120f, tensor.valueAt(1, 2))
        assertEquals(130f, tensor.valueAt(1, 3))

        assertEquals(310f, tensor.valueAt(3, 1))
        assertEquals(320f, tensor.valueAt(3, 2))
        assertEquals(330f, tensor.valueAt(3, 3))

        assertEquals(510f, tensor.valueAt(5, 1))
        assertEquals(520f, tensor.valueAt(5, 2))
        assertEquals(530f, tensor.valueAt(5, 3))

        assertEquals(0f, tensor.valueAt(0, 0))
        assertEquals(20f, tensor.valueAt(2, 0))
        assertEquals(40f, tensor.valueAt(4, 0))
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
    fun `mixed mutable indexing produces writable view`() {
        val tensor = MutableTensor(4, 5, 6) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val view = tensor[
            Index(2),
            Range(1..3),
            Slice(0, 6, 2),
        ]

        assertEquals(listOf(3, 3), view.shape)

        view.setValueAt(
            1,
            2,
            value = 69420f,
        )

        assertEquals(69420f, view.valueAt(1, 2))
        assertEquals(69420f, tensor.valueAt(2, 2, 4))
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

    @Test
    fun `permutation changes axes correctly`() {
        val tensor = Tensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val transposed = tensor.T[2, 0, 1]

        assertEquals(listOf(4, 2, 3), transposed.shape)

        assertEquals(
            tensor.valueAt(1, 2, 3),
            transposed.valueAt(3, 1, 2),
        )
    }

    @Test
    fun `mutable permutation is a view`() {
        val tensor = MutableTensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val transposed = tensor.T[2, 0, 1]

        transposed.setValueAt(
            3,
            1,
            2,
            value = 69420f,
        )

        assertEquals(
            69420f,
            transposed.valueAt(3, 1, 2),
        )

        assertEquals(
            69420f,
            tensor.valueAt(1, 2, 3),
        )
    }

    @Test
    fun `slice after permutation still maps to correct backing element`() {
        val tensor = MutableTensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val view =
            tensor
                .T[2, 0, 1][
                Range(1..3),
                Index(1),
                Slice(0, 3, 2),
            ]

        assertEquals(listOf(3, 2), view.shape)

        assertEquals(101f, view.valueAt(0, 0))
        assertEquals(121f, view.valueAt(0, 1))

        assertEquals(102f, view.valueAt(1, 0))
        assertEquals(122f, view.valueAt(1, 1))

        assertEquals(103f, view.valueAt(2, 0))
        assertEquals(123f, view.valueAt(2, 1))

        view.setValueAt(
            2,
            1,
            value = 999f,
        )

        assertEquals(999f, tensor.valueAt(1, 2, 3))
    }

    @Test
    fun `three dimensional scalar field produces sensible reductions`() {
        val field = Tensor(DEPTH, HEIGHT, WIDTH) { (z, y, x) ->
            val nx =
                (x - WIDTH / 2f) /
                        (WIDTH / 2f)

            val ny =
                (y - HEIGHT / 2f) /
                        (HEIGHT / 2f)

            val nz =
                (z - DEPTH / 2f) /
                        (DEPTH / 2f)

            sqrt(
                nx * nx +
                        ny * ny +
                        nz * nz,
            )
        }

        assertEquals(
            listOf(DEPTH, HEIGHT, WIDTH),
            field.shape,
        )

        assertEquals(3, field.rank)
        assertEquals(
            DEPTH * HEIGHT * WIDTH,
            field.size,
        )

        assertEquals(
            0.071624234f,
            field.min(),
            FLOAT_TOLERANCE,
        )

        assertEquals(
            1.7320508f,
            field.max(),
            FLOAT_TOLERANCE,
        )

        assertEquals(
            0.9619557f,
            field.mean(),
            FLOAT_TOLERANCE,
        )
    }

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

    companion object {
        private const val DEPTH = 21
        private const val HEIGHT = 21
        private const val WIDTH = 41

        private const val FLOAT_TOLERANCE = 0.000001f
    }
}