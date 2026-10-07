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
                val value = result[z, y, x]

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
}