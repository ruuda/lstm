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

  def *(that: Dual): Dual = this.zipWith(that, _ * _, (x, dx, y, dy) => x * dy + y * dx)

  def tanh: Dual = {
    val tanhx = scala.math.tanh(x)
    val dtanhx = 1.0 - tanhx * tanhx
    Dual(tanhx, dxs.map(dx => dx * dtanhx))
  }
}

// A vector in a finite-dimensional real vector space.
// (Not to be confused with scala.collections.immutable.Vector.)
case class Vec(xs: Seq[Double]) {
  private def zipWith(that: Vec, f: (Double, Double) => Double): Vec = {
    val Vec(ys) = that
    require(xs.length == ys.length)
    val zs = xs.zip(ys).map { case (x, y) => f(x, y) }
    Vec(zs)
  }

  def +(that: Vec): Vec = this.zipWith(that, _ + _)

  def ++(that: Vec): Vec = Vec(xs ++ that.xs)

  def pointwiseMul(that: Vec): Vec = this.zipWith(that, _ * _)

  def dot(that: Vec): Double = (this pointwiseMul that).xs.sum

  def map(f: Double => Double): Vec = Vec(xs.map(f))

  def tanh: Vec = this.map(scala.math.tanh)
}

case class Mat(xss: Seq[Seq[Double]]) {
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
}

case class Layer(weight: Mat, offset: Vec) {
  def eval(input: Vec): Vec = weight * input + offset
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
}

object Main {
  def main(args: Array[String]): Unit = {
    println("Hi")
  }
}
