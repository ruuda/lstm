// Copyright 2017 Ruud van Asseldonk

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3. A copy
// of the License is available in the root of the repository.

import scala.util.Random

// A real number, and its derivative with respect to a number of variables.
// Or more precisely: f(x_0, x_1, ..., x_i), (df/dx_0)(x_0, x_1, ..., x_i), ...
case class Dual(x: Double, dxs: Seq[Double]) {

  private def zipWith(that: Dual,
                      f: (Double, Double) => Double,
                      g: (Double, Double, Double, Double) => Double): Dual = {
    val Dual(y, dys) = that
    require(dxs.length == dys.length)
    val z = f(x, y)
    val dzs = dxs.zip(dys).map { case (dx, dy) => g(x, dx, y, dy) }
    Dual(z, dzs)
  }

  def +(that: Dual): Dual = this.zipWith(that, _ + _, (_, dx, _, dy) => dx + dy)
  def -(that: Dual): Dual = this.zipWith(that, _ - _, (_, dx, _, dy) => dx - dy)

  def *(that: Dual): Dual = this.zipWith(that, _ * _, (x, dx, y, dy) => x * dy + y * dx)

  def tanh: Dual = {
    val tanhx = scala.math.tanh(x)
    val dtanhx = 1.0 - tanhx * tanhx
    Dual(tanhx, dxs.map(dx => dx * dtanhx))
  }
}

object Dual {
  def sum(duals: Seq[Dual]): Dual = {
    if (duals.isEmpty) {
      Dual(0.0, Seq.empty)
    } else {
      val zero = Dual(0.0, duals.head.dxs.map(_ => 0.0))
      duals.foldLeft(zero) { case (acc, x) => acc + x }
    }
  }

  def constant(value: Double, total: Int): Dual =
    Dual(value, Seq.range(0, total).map(_ => 0.0))

  // Construct a dual number where its derivative with respect to all variables
  // is 0, except for the one at the given index, for which it is 1.
  def variable(value: Double, index: Int, total: Int): Dual = {
    val dxs = Seq.range(0, total).map(i => if (i == index) { 1.0 } else { 0.0 })
    Dual(value, dxs)
  }
}

// A vector in a finite-dimensional real vector space.
// (Not to be confused with scala.collections.immutable.Vector.)
case class Vec(xs: Seq[Dual]) {
  private def zipWith(that: Vec, f: (Dual, Dual) => Dual): Vec = {
    val Vec(ys) = that
    require(xs.length == ys.length)
    val zs = xs.zip(ys).map { case (x, y) => f(x, y) }
    Vec(zs)
  }

  def +(that: Vec): Vec = this.zipWith(that, _ + _)
  def -(that: Vec): Vec = this.zipWith(that, _ - _)

  def *(that: Dual): Vec = Vec(xs.map(x => x * that))

  def ++(that: Vec): Vec = Vec(xs ++ that.xs)

  def pointwiseMul(that: Vec): Vec = this.zipWith(that, _ * _)

  def dot(that: Vec): Dual = Dual.sum((this pointwiseMul that).xs)

  def map(f: Dual => Dual): Vec = Vec(xs.map(f))

  def tanh: Vec = this.map(x => x.tanh)

  override def toString: String = "[" ++ xs.map(w => f"${w.x}%5.2f").mkString(", ") ++ "]"
}

object Vec {
  def constant(xs: Seq[Double], total: Int): Vec = Vec(xs.map(Dual.constant(_, total)))
}

case class Mat(xss: Seq[Seq[Dual]]) {
  // Every row must have the same number of columns.
  require(xss.map(_.length).max == xss.map(_.length).min)

  // Matrix-vector multiplication. Treats the vector as a column vector,
  // and xss as the rows of the matrix.
  def *(that: Vec): Vec = {
    val Vec(ys) = that
    require(xss.forall(_.length == ys.length))
    val zs = xss.map(xs => Vec(xs) dot that)
    Vec(zs)
  }

  def *(that: Dual): Mat = Mat(xss.map(row => row.map(x => x * that)))

  def -(that: Mat): Mat = {
    val Mat(yss) = that
    require(xss.length == yss.length)
    require(xss.zipWithIndex.forall { case (row, i) => row.length == yss(i).length })
    val zss = xss.zip(yss).map {
      case (xs, ys) => xs.zip(ys).map { case (x, y) => x - y }
    }
    Mat(zss)
  }

  override def toString: String = "[" + xss.map(
    row => "[" ++ row.map(w => f"${w.x}%5.2f").mkString(", ") ++ "]"
  ).mkString("\n ") ++ "]"
}

case class Layer(weight: Mat, offset: Vec) {
  def eval(input: Vec): Vec = weight * input + offset

  def *(that: Dual): Layer = Layer(weight * that, offset * that)

  def -(that: Layer): Layer = Layer(weight - that.weight, offset - that.offset)
}

object Layer {
  // Build a layer of the given size with random weights. The weights are
  // variables for the gradient, starting at the given index.
  def build(random: Random,
            inputLen: Int,
            outputLen: Int,
            gradientIndex: Int,
            gradientTotal: Int): Layer = {
    val rows = Seq.range(0, outputLen).map {
      i => Seq.range(0, inputLen).map {
        j =>
          // Generate a random weight between -1 and 1.
          val weight = random.nextDouble() * 2.0 - 1.0
          val index = gradientIndex + i * inputLen + j
          Dual.variable(weight, index, gradientTotal)
      }
    }
    val offsets = Seq.range(0, outputLen).map {
      i =>
        val weight = random.nextDouble() * 2.0 - 1.0
        val index = gradientIndex + inputLen * outputLen + i
        Dual.variable(weight, index, gradientTotal)
    }
    Layer(Mat(rows), Vec(offsets))
  }

  def fromGradient(gradient: Seq[Double],
                   inputLen: Int,
                   outputLen: Int,
                   gradientIndex: Int,
                   gradientTotal: Int): Layer = {
    val rows = Seq.range(0, outputLen).map {
      i => Seq.range(0, inputLen).map {
        j => Dual.constant(gradient(gradientIndex + i * inputLen + j), gradientTotal)
      }
    }
    val offsets = Seq.range(0, outputLen).map {
      i => Dual.constant(gradient(gradientIndex + inputLen * outputLen + i), gradientTotal)
    }
    Layer(Mat(rows), Vec(offsets))
  }
}

// An implementation of an LSTM layer based on
// https://colah.github.io/posts/2015-08-Understanding-LSTMs/
case class Lstm(forgetGate: Layer, inputGate: Layer, candidate: Layer, output: Layer) {

  // Returns (newState, output).
  def eval(state: Vec, prevOutput: Vec, input: Vec): (Vec, Vec) = {
    val fullInput = prevOutput ++ input

    // TODO: Do I need the tanh here?
    val forgetMask = forgetGate.eval(fullInput).tanh
    val updateMask = inputGate.eval(fullInput).tanh
    val candidates = candidate.eval(fullInput).tanh

    val newState = (state pointwiseMul forgetMask) + (candidates pointwiseMul updateMask)

    // TODO: Do I need the tanh after the output eval?
    val out = output.eval(fullInput).tanh pointwiseMul newState.tanh

    (newState, out)
  }

  def *(that: Dual): Lstm = Lstm(
    forgetGate * that,
    inputGate * that,
    candidate * that,
    output * that
  )

  def -(that: Lstm): Lstm = Lstm(
    forgetGate - that.forgetGate,
    inputGate - that.inputGate,
    candidate - that.candidate,
    output - that.output
  )
}

object Lstm {
  def getGradientLen(inputLen: Int, outputLen: Int): Int =
    4 * ((inputLen + outputLen) * outputLen + outputLen)

  def build(random: Random,
            inputLen: Int,
            outputLen: Int,
            gradientIndex: Int,
            gradientTotal: Int): Lstm = {
    val fullInputLen = outputLen + inputLen;
    val layerLen = fullInputLen * outputLen + outputLen;
    require(gradientTotal >= layerLen * 4,
      s"Need at least ${4 * layerLen} variables for weights, but only $gradientTotal given.")
    val forgetGate = Layer.build(random, fullInputLen, outputLen, gradientIndex + 0 * layerLen, gradientTotal)
    val inputGate = Layer.build(random, fullInputLen, outputLen, gradientIndex + 1 * layerLen, gradientTotal)
    val candidate = Layer.build(random, fullInputLen, outputLen, gradientIndex + 2 * layerLen, gradientTotal)
    val output = Layer.build(random, fullInputLen, outputLen, gradientIndex + 3 * layerLen, gradientTotal)
    Lstm(forgetGate, inputGate, candidate, output)
  }

  def fromGradient(gradient: Seq[Double],
                   inputLen: Int,
                   outputLen: Int,
                   gradientIndex: Int,
                   gradientTotal: Int): Lstm = {
    val fullInputLen = inputLen + outputLen;
    val layerLen = fullInputLen * outputLen + outputLen;
    require(gradientTotal >= layerLen * 4,
      s"Need at least ${4 * layerLen} variables for weights, but only $gradientTotal given.")
    val forgetGate = Layer.fromGradient(gradient, fullInputLen, outputLen, gradientIndex + 0 * layerLen, gradientTotal)
    val inputGate = Layer.fromGradient(gradient, fullInputLen, outputLen, gradientIndex + 1 * layerLen, gradientTotal)
    val candidate = Layer.fromGradient(gradient, fullInputLen, outputLen, gradientIndex + 2 * layerLen, gradientTotal)
    val output = Layer.fromGradient(gradient, fullInputLen, outputLen, gradientIndex + 3 * layerLen, gradientTotal)
    Lstm(forgetGate, inputGate, candidate, output)
  }
}


object Main {
  def main(args: Array[String]): Unit = {
    val random = new Random()
    val glen = Lstm.getGradientLen(5, 3)
    var lstm = Lstm.build(random, 5, 3, 0, glen)
    for (i <- Seq.range(0, 100)) {
      // Pick a few silly examples that we learn at every iteration.
      val state = Vec.constant(Seq(0.0, 0.0, 0.0), glen)

      val inputs = Seq(
        Vec.constant(Seq(1.0, 0.0, 0.0, 0.0, 2.0), glen),
        Vec.constant(Seq(0.0, 1.0, 0.0, 1.0, 0.0), glen),
        Vec.constant(Seq(0.0, 0.0, 1.0, 0.0, 0.0), glen)
      )

      val expected = Seq(
        Vec.constant(Seq(0.3, 0.3, 0.3), glen),
        Vec.constant(Seq(0.0, 0.2, 0.2), glen),
        Vec.constant(Seq(0.0, 0.0, 0.1), glen)
      )

      val actual = inputs.map(input => lstm.eval(state, state, input)._2)
      val error = Dual.sum(actual.zip(expected).map {
        case (a, e) =>
          val diff = a - e
          diff dot diff
      })

      println(s"Error at iteration $i: ${error.x}.")

      // Do a gradient descent: we have the gradient of the error with respect
      // to all the weights, so now update the weights such that the error will
      // decrease.
      val correction = Lstm.fromGradient(error.dxs, 5, 3, 0, glen)
      lstm = lstm - correction * Dual.constant(0.1, glen)
    }
  }
}
