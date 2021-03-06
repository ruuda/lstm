// Copyright 2017 Ruud van Asseldonk

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3. A copy
// of the License is available in the root of the repository.

import scala.util.Random

// A real number, and its derivative with respect to a number of variables.
// Or more precisely: f(x_0, x_1, ..., x_i), (df/dx_0)(x_0, x_1, ..., x_i), ...
case class Dual(x: Double, dxs: Array[Double]) {

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

  // Tanh but scaled and translated to fit [0, 1] instead of [-1, 1].
  def sigmoid: Dual = {
    val t = this.tanh
    Dual(t.x * 0.5 + 0.5, t.dxs.map(dx => dx * 0.5))
  }
}

object Dual {
  def sum(duals: Array[Dual]): Dual = {
    if (duals.isEmpty) {
      Dual(0.0, Array.empty)
    } else {
      val zero = Dual(0.0, duals.head.dxs.map(_ => 0.0))
      duals.foldLeft(zero) { case (acc, x) => acc + x }
    }
  }

  def constant(value: Double, total: Int): Dual =
    Dual(value, Array.range(0, total).map(_ => 0.0))

  // Construct a dual number where its derivative with respect to all variables
  // is 0, except for the one at the given index, for which it is 1.
  def variable(value: Double, index: Int, total: Int): Dual = {
    val dxs = Array.range(0, total).map(i => if (i == index) { 1.0 } else { 0.0 })
    Dual(value, dxs)
  }
}

// A vector in a finite-dimensional real vector space.
// (Not to be confused with scala.collections.immutable.Vector.)
case class Vec(xs: Array[Dual]) {
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

  def sigmoid: Vec = this.map(x => x.sigmoid)

  override def toString: String = "[" ++ xs.map(w => f"${w.x}%5.2f").mkString(", ") ++ "]"
}

object Vec {
  def constant(xs: Array[Double], total: Int): Vec = Vec(xs.map(Dual.constant(_, total)))
}

case class Mat(xss: Array[Array[Dual]]) {
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
    val rows = Array.range(0, outputLen).map {
      i => Array.range(0, inputLen).map {
        j =>
          // Generate a random weight between -1 and 1.
          val weight = random.nextDouble() * 2.0 - 1.0
          val index = gradientIndex + i * inputLen + j
          Dual.variable(weight, index, gradientTotal)
      }
    }
    val offsets = Array.range(0, outputLen).map {
      i =>
        val weight = random.nextDouble() * 2.0 - 1.0
        val index = gradientIndex + inputLen * outputLen + i
        Dual.variable(weight, index, gradientTotal)
    }
    Layer(Mat(rows), Vec(offsets))
  }

  def fromGradient(gradient: Array[Double],
                   inputLen: Int,
                   outputLen: Int,
                   gradientIndex: Int,
                   gradientTotal: Int): Layer = {
    val rows = Array.range(0, outputLen).map {
      i => Array.range(0, inputLen).map {
        j => Dual.constant(gradient(gradientIndex + i * inputLen + j), gradientTotal)
      }
    }
    val offsets = Array.range(0, outputLen).map {
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

    val forgetMask = forgetGate.eval(fullInput).sigmoid
    val updateMask = inputGate.eval(fullInput).sigmoid
    val candidates = candidate.eval(fullInput).tanh

    val newState = (state pointwiseMul forgetMask) + (candidates pointwiseMul updateMask)

    val out = output.eval(fullInput).sigmoid pointwiseMul newState.tanh

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

  def fromGradient(gradient: Array[Double],
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

class LetterPredictor {
  // Use two LSTM cells, one with 26 inputs (one for every letter) and 15
  // outputs, and one with 15 inputs and and 26 outputs (1 for every letter).
  private val glen0 = Lstm.getGradientLen(26, 15)
  private val glen1 = Lstm.getGradientLen(15, 26)
  private val glen = glen0 + glen1
  println(s"Variables: $glen.")

  private val random = new Random()
  private var lstm0 = Lstm.build(random, 26, 15, 0, glen)
  private var lstm1 = Lstm.build(random, 15, 26, glen0, glen)

  // Input used to seed the network. We feed this as initial internal state, but
  // also before the first input character.
  private val zero26 = Vec.constant(Array.range(0, 26).map(_ => 0.0), glen)
  private val zero15 = Vec.constant(Array.range(0, 15).map(_ => 0.0), glen)

  // Convert a lowercase Latin letter to an input or output for the network. The
  // input has 26 values, one for every letter. They are all 0, except for the
  // one at the letter index, which is 1.
  private def letterToVec(c: Char): Vec = {
    val index = c - 'a'
    val coords = Array.range(0, 26).map(i => if (i == index) { 1.0 } else { 0.0 })
    Vec.constant(coords, glen)
  }

  // Return the letter for which confidence was highest, if the confidence
  // was more than 0.1, or a space otherwise.
  private def vecToLetter(v: Vec): Char = {
    require(v.xs.length == 26)
    val (confidence, i) = v.xs.zipWithIndex.maxBy { case (c, i) => c.x }
    if (confidence.x > 0.1) {
      ('a' + i).toChar
    } else {
      ' '
    }
  }

  // Feed the entire string into the network, letter by letter, and return
  // the output of the last iteration converted into a letter.
  def predict(str: String): Char = {
    require(str.forall(_.isLetter) && str.forall(_.isLower),
      "Input must be lowercase letters only.")

    // Seed the state and previous output, by proding an input of all zeros,
    // with a state and previous output of all zeros too.
    val (istate0, ioutput0) = lstm0.eval(zero15, zero15, zero26)
    val (istate1, ioutput1) = lstm1.eval(zero26, zero26, ioutput0)

    // Feed the string into the network, letter by letter.
    val (_, _, _, finalOutput) = str.foldLeft((istate0, ioutput0, istate1, ioutput1)) {
      case ((state0, output0, state1, output1), c) =>
        val (newState0, newOutput0) = lstm0.eval(state0, output0, letterToVec(c))
        val (newState1, newOutput1) = lstm1.eval(state1, output1, newOutput0)
        (newState0, newOutput0, newState1, newOutput1)
    }

    vecToLetter(finalOutput)
  }

  // Predict every letter in the string, and sum up all the errors.
  private def evalStringError(str: String): Dual = {
    require(str.forall(_.isLetter) && str.forall(_.isLower),
      "Input must be lowercase letters only.")

    print(s"  Evaluating string: ")
    print(str.head)

    // Seed the network with zeros as initial state, previous output, and
    // current input. Then feed in the first letter.
    val (istate0, ioutput0) = lstm0.eval(zero15, zero15, zero26)
    val (istate1, ioutput1) = lstm1.eval(zero26, zero26, ioutput0)
    var (state0, output0) = lstm0.eval(istate0, ioutput0, letterToVec(str.head))
    var (state1, output1) = lstm1.eval(istate1, ioutput1, output0)

    var error = Dual.constant(0.0, glen)
    var prevChar = str.head
    var prediction = str.substring(0, 1)

    // Loop over the remaining letters, and compare the network output with the
    // desired output to determine the error. The desired output after the last
    // letter is a space, which encodes as the zero vector as desired output.
    for (c <- (str.tail ++ " ")) {
      print(c)
      val nextInput = letterToVec(c)
      val diff = output1 - nextInput
      error = error + (diff dot diff)
      prediction += vecToLetter(output1)

      if (c != ' ') {
        val (s0, o0) = lstm0.eval(state0, output0, nextInput)
        val (s1, o1) = lstm1.eval(state1, output1, o0)
        state0 = s0
        state1 = s1
        output0 = o0
        output1 = o1
      }
    }

    println(f"[$prediction] => error: ${error.x / str.length}%.3f.")

    error
  }

  def learnStrings(strs: Array[String], rate: Double): Double = {
    val count = strs.map(_.length).sum
    val error = Dual.sum(strs.map(evalStringError)) * Dual.constant(1.0 / count.toDouble, glen)
    println(f"Error per letter: ${error.x}%.3f")

    val correction0 = Lstm.fromGradient(error.dxs, 26, 15, 0, glen)
    val correction1 = Lstm.fromGradient(error.dxs, 15, 26, glen0, glen)
    lstm0 = lstm0 - correction0 * Dual.constant(rate, glen)
    lstm1 = lstm1 - correction1 * Dual.constant(rate, glen)

    error.x
  }
}


object Main {
  def main(args: Array[String]): Unit = {
    // A list of words that exists on my machine as part of the cracklib
    // package, which is a dependency of pam, and thereby of basically
    // everything.
    val words = io.Source.fromFile("/usr/share/dict/cracklib-small")
      .getLines
      .map(word => word.map(_.toLower).filter(_.isLetter))
      .toSeq
    val wordsRandom = new Random().shuffle(words)

    println("Constructing LetterPredictor ...")
    val predictor = new LetterPredictor()
    println("Initiating learning process ...")

    // Learn groups of 5 words at a time.
    val wordGroups = wordsRandom.grouped(5)

    var delta = 1.0
    var error = 100.0
    while (delta > 0.001) {
      val newError = predictor.learnStrings(wordGroups.next().toArray, 0.1)
      delta = (error - newError).abs
      error = newError
    }

    println(s"Prediction of 'fo_': ${predictor.predict("fo")}.")
    println(s"Prediction of 'ba_': ${predictor.predict("ba")}.")
    println(s"Prediction of 'fi_': ${predictor.predict("fi")}.")
    println(s"Prediction of 'fiz_': ${predictor.predict("fiz")}.")
  }
}
