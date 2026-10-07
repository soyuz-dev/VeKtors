package org.soyuz

import org.soyuz.vektors.MutableTensor
import org.soyuz.vektors.Tensor
import kotlin.math.sqrt

private const val DEPTH = 21
private const val HEIGHT = 21
private const val WIDTH = 41

fun main() {
    val field = Tensor(DEPTH, HEIGHT, WIDTH) { (z, y, x) ->
        val nx = (x - WIDTH / 2f) / (WIDTH / 2f)
        val ny = (y - HEIGHT / 2f) / (HEIGHT / 2f)
        val nz = (z - DEPTH / 2f) / (DEPTH / 2f)

        sqrt(nx * nx + ny * ny + nz * nz)
    }

    println("Shape: ${field.shape}")
    println("Rank: ${field.rank}")
    println("Size: ${field.size}")
    println("Min: ${field.min()}")
    println("Max: ${field.max()}")
    println("Mean: ${field.mean()}")
    println()

    val density = field.map { distance ->
        1f - distance
    }

    val mutable = MutableTensor(density)

    // Exercise mutable tensor operations too.
    mutable *= 1.5f

    val result = density + mutable

    for (z in 0 until DEPTH) {
        println("z = $z")

        for (y in 0 until HEIGHT) {
            for (x in 0 until WIDTH) {
                val value = result.valueAt(z, y, x)

                val char = when {
                    value > 2.5f -> '█'
                    value > 2.0f -> '▓'
                    value > 1.5f -> '▒'
                    value > 1.0f -> '░'
                    else -> ' '
                }

                print(char)
            }

            println()
        }

        println()
    }

    val a = Tensor(2, 3) { (row, column) ->
        (row * 10 + column).toFloat()
    }

    val b = Tensor(3) { (column) ->
        (column + 1).toFloat()
    }

    val c = a + b

    for (row in 0 until 2) {
        for (column in 0 until 3) {
            print("${c.valueAt(row, column)} ")
        }

        println()
    }

    println()

    // New indexing semantics.
    println("a.shape = ${a.shape}")
    println("a[0].shape = ${a[0].shape}")
    println("a[0][1].shape = ${a[0][1].shape}")

    println("a[0].sum() = ${a[0].sum()}")
    println("a[1].sum() = ${a[1].sum()}")

    println("a.valueAt(1, 2) = ${a.valueAt(1, 2)}")
    println("a[1][2].valueAt() = ${a[1][2].valueAt()}")

    println()

    // Explicit scalar tensor.
    val scalar = Tensor.scalar(42f)

    println("scalar.shape = ${scalar.shape}")
    println("scalar.rank = ${scalar.rank}")
    println("scalar.size = ${scalar.size}")
    println("scalar.valueAt() = ${scalar.valueAt()}")
    println("scalar.sum() = ${scalar.sum()}")

    println()

    // Mutable scalar access.
    val mutableTest = MutableTensor(2, 3) { (row, column) ->
        (row * 10 + column).toFloat()
    }

    println(
        "Before: ${mutableTest.valueAt(1, 2)}"
    )

    mutableTest.setValueAt(
        1,
        2,
        value = 69f,
    )

    println(
        "After: ${mutableTest.valueAt(1, 2)}"
    )

    val viewTest = MutableTensor(2, 3) { (row, column) ->
        (row * 10 + column).toFloat()
    }

    val row = viewTest[1]

    println(row.shape)
    println(row.valueAt(2))
    println(viewTest.valueAt(1, 2))

    row *= 2f

    println(row.valueAt(2))
    println(viewTest.valueAt(1, 2))

    val scalarView = viewTest[1][2]

    println(scalarView.shape)
    println(scalarView.valueAt())

    scalarView *= 10f

    println(scalarView.valueAt())
    println(viewTest.valueAt(1, 2))
}