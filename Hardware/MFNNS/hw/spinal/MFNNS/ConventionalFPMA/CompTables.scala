package MFNNS.ConventionalFPMA

import spinal.core._
import scala.language.postfixOps

case class CompTable2D(name: String, data: Vector[Vector[Int]], compWidthOverride: Option[Int] = None) {
  require(data.nonEmpty, s"$name must not be empty")
  require(data.head.nonEmpty, s"$name must not have empty rows")
  require(data.forall(_.length == data.head.length), s"$name must be rectangular")
  require(isPow2(data.length), s"$name row count must be a power of two")
  require(isPow2(data.head.length), s"$name column count must be a power of two")
  require(data.flatten.forall(_ >= 0), s"$name only supports non-negative values")

  val rows: Int = data.length
  val cols: Int = data.head.length
  val aIdxWidth: Int = log2Up(rows)
  val wIdxWidth: Int = log2Up(cols)
  val idxWidth: Int = aIdxWidth + wIdxWidth
  val maxValue: Int = data.flatten.max
  val compWidth: Int = compWidthOverride.getOrElse(math.max(1, log2Up(maxValue + 1)))
  val flatData: Vector[Int] = data.flatten
  val defaultModuleName: String = s"Comp_M10_${rows}x${cols}"

  private def isPow2(x: Int): Boolean = x > 0 && ((x & (x - 1)) == 0)
}

object CompTables {
  val E5_M10_16x16: CompTable2D = CompTable2D(
    name = "E5_M10_16x16",
    data = Vector(
      Vector( 1,  3,   5,   7,   9,  11,  13,  15,  17, 19, 21, 23, 25, 27, 27, 13),
      Vector( 3,  9,  15,  21,  27,  33,  39,  45,  51, 57, 63, 69, 74, 68, 44, 15),
      Vector( 5, 15,  25,  35,  45,  55,  65,  75,  85, 95,105,110, 94, 68, 41, 14),
      Vector( 7, 21,  35,  49,  63,  77,  91, 105, 119,132,134,113, 88, 63, 38, 13),
      Vector( 9, 27,  45,  63,  81,  99, 117, 135, 151,149,127,104, 81, 58, 35, 12),
      Vector(11, 33,  55,  77,  99, 121, 143, 162, 157,137,116, 95, 74, 53, 32, 11),
      Vector(13, 39,  65,  91, 117, 143, 166, 162, 143,124,105, 86, 67, 48, 29, 10),
      Vector(15, 45,  75, 105, 135, 162, 162, 145, 128,111, 94, 77, 60, 43, 26,  9),
      Vector(17, 51,  85, 119, 151, 157, 143, 128, 113, 98, 83, 68, 53, 38, 23,  8),
      Vector(19, 57,  95, 132, 149, 137, 124, 111,  98, 85, 72, 59, 46, 33, 20,  7),
      Vector(21, 63, 105, 134, 127, 116, 105,  94,  83, 72, 61, 50, 39, 28, 17,  6),
      Vector(23, 69, 110, 113, 104,  95,  86,  77,  68, 59, 50, 41, 32, 23, 14,  5),
      Vector(25, 74,  94,  88,  81,  74,  67,  60,  53, 46, 39, 32, 25, 18, 11,  4),
      Vector(27, 68,  68,  63,  58,  53,  48,  43,  38, 33, 28, 23, 18, 13,  8,  3),
      Vector(27, 44,  41,  38,  35,  32,  29,  26,  23, 20, 17, 14, 11,  8,  5,  2),
      Vector(13, 15,  14,  13,  12,  11,  10,   9,   8,  7,  6,  5,  4,  3,  2,  0)
    ),
    compWidthOverride = Some(8)
  )

  val E5_M10_8x8: CompTable2D = CompTable2D(
    name = "E5_M10_8x8",
    data = Vector(
      Vector( 4,  12,  20,  28,  36,  44, 48, 24),
      Vector(12,  36,  60,  84, 108, 115, 78, 26),
      Vector(20,  60, 100, 139, 148, 110, 66, 22),
      Vector(28,  84, 139, 158, 126,  90, 54, 18),
      Vector(36, 108, 148, 126,  98,  70, 42, 14),
      Vector(44, 115, 110,  90,  70,  50, 30, 10),
      Vector(48,  78,  66,  54,  42,  30, 18,  6),
      Vector(24,  26,  22,  18,  14,  10,  6,  2)
    ),
    compWidthOverride = Some(8)
  )

  val E5_M10_4x4: CompTable2D = CompTable2D(
    name = "E5_M10_4x4",
    data = Vector(
      Vector(16,  48,  76, 44),
      Vector(48, 134, 119, 40),
      Vector(76, 119,  72, 24),
      Vector(44,  40,  24,  8)
    ),
    compWidthOverride = Some(8)
  )

  val E5_M10_2x2: CompTable2D = CompTable2D(
    name = "E5_M10_2x2",
    data = Vector(
      Vector(61, 70),
      Vector(70, 32)
    ),
    compWidthOverride = Some(8)
  )

  val all: Seq[CompTable2D] = Seq(
    E5_M10_16x16,
    E5_M10_8x8,
    E5_M10_4x4,
    E5_M10_2x2
  )

  val byName: Map[String, CompTable2D] = all.map(t => t.name -> t).toMap
  val byShape: Map[(Int, Int), CompTable2D] = all.map(t => (t.rows, t.cols) -> t).toMap

  def square(size: Int): CompTable2D = byShape((size, size))
}
