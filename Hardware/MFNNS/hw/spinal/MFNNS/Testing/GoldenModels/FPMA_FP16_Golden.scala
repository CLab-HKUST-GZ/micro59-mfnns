package MFNNS.Testing.GoldenModels

/** Bit-exact software model of `MFNNS.ConventionalFPMA.FPMA_FP16`.
  *
  * This models the encoded exponent/mantissa additions and their fixed-width
  * wraparound. It is an implementation oracle, not an IEEE-754 multiplier.
  */
object FPMA_FP16_Golden {

  val InputWidth: Int = 16
  val MagnitudeWidth: Int = 15

  private val InputMask = 0xffff
  private val MagnitudeMask = 0x7fff
  private val MantissaIndexMask = 0x0f
  private val MantissaIndexShift = 6

  // RTL concatenation: 6'b110001 ## 2'b00 ## compensation[7:0].
  private val NegativeBiasPrefix = 0xc400

  final case class Trace(
      inputA: Int,
      inputW: Int,
      aMagnitude: Int,
      wMagnitude: Int,
      aMantissaMsb: Int,
      wMantissaMsb: Int,
      compensation: Int,
      correction: Int,
      stage1Sum: Int,
      stage2Sum: Int,
      resultSign: Int,
      resultMagnitude: Int,
      result: Int
  )

  /** Fast result-only path for large simulation sweeps. */
  def calculate(aBits: Int, wBits: Int): Int = {
    requireFP16Bits(aBits, "A")
    requireFP16Bits(wBits, "W")

    val aMagnitude = aBits & MagnitudeMask
    val wMagnitude = wBits & MagnitudeMask
    val aMantissaMsb = (aBits >>> MantissaIndexShift) & MantissaIndexMask
    val wMantissaMsb = (wBits >>> MantissaIndexShift) & MantissaIndexMask
    val compensation = CompensationTableGolden.comp16x16(aMantissaMsb, wMantissaMsb)
    val correction = NegativeBiasPrefix | compensation
    val stage1Sum = (aMagnitude + wMagnitude) & InputMask
    val stage2Sum = (stage1Sum + correction) & InputMask
    val resultSign = ((aBits ^ wBits) >>> 15) & 0x1

    (resultSign << MagnitudeWidth) | (stage2Sum & MagnitudeMask)
  }

  def evaluate(aBits: Int, wBits: Int): Trace = {
    requireFP16Bits(aBits, "A")
    requireFP16Bits(wBits, "W")

    val aMagnitude = aBits & MagnitudeMask
    val wMagnitude = wBits & MagnitudeMask
    val aMantissaMsb = (aBits >>> MantissaIndexShift) & MantissaIndexMask
    val wMantissaMsb = (wBits >>> MantissaIndexShift) & MantissaIndexMask
    val compensation = CompensationTableGolden.comp16x16(aMantissaMsb, wMantissaMsb)
    val correction = NegativeBiasPrefix | compensation

    // Both RTL adders are 16 bits wide and discard carry-out independently.
    val stage1Sum = (aMagnitude + wMagnitude) & InputMask
    val stage2Sum = (stage1Sum + correction) & InputMask

    val resultSign = ((aBits ^ wBits) >>> 15) & 0x1
    val resultMagnitude = stage2Sum & MagnitudeMask
    val result = (resultSign << MagnitudeWidth) | resultMagnitude

    Trace(
      inputA = aBits,
      inputW = wBits,
      aMagnitude = aMagnitude,
      wMagnitude = wMagnitude,
      aMantissaMsb = aMantissaMsb,
      wMantissaMsb = wMantissaMsb,
      compensation = compensation,
      correction = correction,
      stage1Sum = stage1Sum,
      stage2Sum = stage2Sum,
      resultSign = resultSign,
      resultMagnitude = resultMagnitude,
      result = result
    )
  }

  private def requireFP16Bits(bits: Int, operandName: String): Unit = {
    require(bits >= 0 && bits <= InputMask, f"$operandName input must be an unsigned 16-bit value: 0x$bits%x")
  }
}
