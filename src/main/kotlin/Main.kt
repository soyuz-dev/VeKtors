package org.soyuz.vektors.demo

import org.soyuz.vektors.Matrix
import org.soyuz.vektors.Vector
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

private const val WIDTH = 80
private const val HEIGHT = 30

private val vertices = listOf(
    Vector(-1f, -1f, -1f),
    Vector( 1f, -1f, -1f),
    Vector( 1f,  1f, -1f),
    Vector(-1f,  1f, -1f),
    Vector(-1f, -1f,  1f),
    Vector( 1f, -1f,  1f),
    Vector( 1f,  1f,  1f),
    Vector(-1f,  1f,  1f),
)

private val edges = listOf(
    0 to 1,
    1 to 2,
    2 to 3,
    3 to 0,

    4 to 5,
    5 to 6,
    6 to 7,
    7 to 4,

    0 to 4,
    1 to 5,
    2 to 6,
    3 to 7,
)

fun main() {
    var angle = 0f

    print("\u001b[2J")

    while (true) {
        val rotation =
            rotationY(angle) *
                    rotationX(angle * 0.7f) *
                    rotationZ(angle * 0.3f)

        val projected = vertices
            .map { rotation * it }
            .map { project(it) }

        val buffer =
            Array(HEIGHT) {
                CharArray(WIDTH) { ' ' }
            }

        for ((start, end) in edges) {
            val a = projected[start]
            val b = projected[end]

            drawLine(
                buffer,
                a.first,
                a.second,
                b.first,
                b.second,
            )
        }

        print("\u001b[H")

        for (row in buffer) {
            println(row.concatToString())
        }

        angle += 0.04f

        Thread.sleep(30)
    }
}

private fun rotationX(angle: Float): Matrix {
    val c = cos(angle)
    val s = sin(angle)

    return Matrix(
        floatArrayOf(
            1f, 0f, 0f,
        ),
        floatArrayOf(
            0f, c, -s,
        ),
        floatArrayOf(
            0f, s, c,
        ),
    )
}

private fun rotationY(angle: Float): Matrix {
    val c = cos(angle)
    val s = sin(angle)

    return Matrix(
        floatArrayOf(
            c, 0f, s,
        ),
        floatArrayOf(
            0f, 1f, 0f,
        ),
        floatArrayOf(
            -s, 0f, c,
        ),
    )
}

private fun rotationZ(angle: Float): Matrix {
    val c = cos(angle)
    val s = sin(angle)

    return Matrix(
        floatArrayOf(
            c, -s, 0f,
        ),
        floatArrayOf(
            s, c, 0f,
        ),
        floatArrayOf(
            0f, 0f, 1f,
        ),
    )
}

private fun project(point: Vector): Pair<Int, Int> {
    val distance = 4f
    val z = point[2] + distance

    val perspective = 1f / z

    // Terminal characters are taller than they are wide,
    // so X gets a larger scale.
    val x =
        WIDTH / 2f +
                point[0] * perspective * 40f

    val y =
        HEIGHT / 2f -
                point[1] * perspective * 20f

    return x.roundToInt() to y.roundToInt()
}

private fun drawLine(
    buffer: Array<CharArray>,
    x0: Int,
    y0: Int,
    x1: Int,
    y1: Int,
) {
    var x = x0
    var y = y0

    val dx = kotlin.math.abs(x1 - x0)
    val dy = kotlin.math.abs(y1 - y0)

    val sx = if (x0 < x1) 1 else -1
    val sy = if (y0 < y1) 1 else -1

    var error = dx - dy

    while (true) {
        if (
            x in 0 until WIDTH &&
            y in 0 until HEIGHT
        ) {
            buffer[y][x] = '.'
        }

        if (x == x1 && y == y1) {
            break
        }

        val twiceError = 2 * error

        if (twiceError > -dy) {
            error -= dy
            x += sx
        }

        if (twiceError < dx) {
            error += dx
            y += sy
        }
    }
}