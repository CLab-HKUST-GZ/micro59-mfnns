package MFNNS.Testing.GoldenModels

/** Independent, immutable compensation-table oracle used by the MFNNS tests.
  *
  * The values are intentionally frozen here instead of importing the RTL-side
  * `MFNNS.ConventionalFPMA.CompTables`. Keeping the oracle independent allows a
  * test to detect accidental changes to the production table.
  */
object CompensationTableGolden {

  val TableSize: Int = 16
  val CompensationWidth: Int = 8

  /** FP16 E5M10 compensation table indexed as `(aMantissaMsb, wMantissaMsb)`. */
  val E5M10_16x16: Vector[Vector[Int]] = Vector(
    Vector(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 27, 13),
    Vector(3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 74, 68, 44, 15),
    Vector(5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 110, 94, 68, 41, 14),
    Vector(7, 21, 35, 49, 63, 77, 91, 105, 119, 132, 134, 113, 88, 63, 38, 13),
    Vector(9, 27, 45, 63, 81, 99, 117, 135, 151, 149, 127, 104, 81, 58, 35, 12),
    Vector(11, 33, 55, 77, 99, 121, 143, 162, 157, 137, 116, 95, 74, 53, 32, 11),
    Vector(13, 39, 65, 91, 117, 143, 166, 162, 143, 124, 105, 86, 67, 48, 29, 10),
    Vector(15, 45, 75, 105, 135, 162, 162, 145, 128, 111, 94, 77, 60, 43, 26, 9),
    Vector(17, 51, 85, 119, 151, 157, 143, 128, 113, 98, 83, 68, 53, 38, 23, 8),
    Vector(19, 57, 95, 132, 149, 137, 124, 111, 98, 85, 72, 59, 46, 33, 20, 7),
    Vector(21, 63, 105, 134, 127, 116, 105, 94, 83, 72, 61, 50, 39, 28, 17, 6),
    Vector(23, 69, 110, 113, 104, 95, 86, 77, 68, 59, 50, 41, 32, 23, 14, 5),
    Vector(25, 74, 94, 88, 81, 74, 67, 60, 53, 46, 39, 32, 25, 18, 11, 4),
    Vector(27, 68, 68, 63, 58, 53, 48, 43, 38, 33, 28, 23, 18, 13, 8, 3),
    Vector(27, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2),
    Vector(13, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0)
  )

  /** Frozen 1D table used by the square-specialized FPMA primitive. */
  val E5M10_Square16: Vector[Int] = Vector(
    1, 9, 25, 49, 81, 121, 166, 145, 113, 85, 61, 41, 25, 13, 5, 0
  )

  require(E5M10_16x16.length == TableSize, s"Expected $TableSize compensation rows")
  require(E5M10_16x16.forall(_.length == TableSize), s"Expected $TableSize compensation columns")
  require(
    E5M10_16x16.flatten.forall(value => value >= 0 && value < (1 << CompensationWidth)),
    s"Every compensation value must fit in $CompensationWidth bits"
  )
  require(E5M10_16x16 == E5M10_16x16.transpose, "The generic FPMA compensation table must be symmetric")
  require(
    E5M10_Square16 == Vector.tabulate(TableSize)(index => E5M10_16x16(index)(index)),
    "The square compensation table must equal the diagonal of the generic table"
  )

  def comp16x16(aMantissaMsb: Int, wMantissaMsb: Int): Int = {
    require(validIndex(aMantissaMsb), s"A compensation index out of range: $aMantissaMsb")
    require(validIndex(wMantissaMsb), s"W compensation index out of range: $wMantissaMsb")
    E5M10_16x16(aMantissaMsb)(wMantissaMsb)
  }

  def squareComp(mantissaMsb: Int): Int = {
    require(validIndex(mantissaMsb), s"Square compensation index out of range: $mantissaMsb")
    E5M10_Square16(mantissaMsb)
  }

  private def validIndex(index: Int): Boolean = index >= 0 && index < TableSize
}
