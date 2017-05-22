
case class Vector(xs: Seq[Float]) {
  def +(that: Vector): Vector = {
    val Vector(ys) = that
    require(xs.length == ys.length)
    val zs = xs.zip(ys).map { case (x, y) => x + y }
    Vector(zs)
  }

  def dot(that: Vector): Float = {
    val Vector(ys) = that
    require(xs.length == ys.length)
    xs.zip(ys).foldLeft(0.0f) { case (acc, (x, y)) => acc + x * y }
  }
}

case class Matrix(xss: Seq[Seq[Float]])

object Main {
  def main(args: Array[String]): Unit = {
    println("Hi")
  }
}
