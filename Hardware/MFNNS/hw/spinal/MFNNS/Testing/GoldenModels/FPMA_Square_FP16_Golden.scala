package MFNNS.Testing.GoldenModels

/** Bit-exact software model of `MFNNS.SquareFPMA.FPMA_Square_FP16`.
  *
  * The primitive implements the square-specialized encoded-domain operation
  * from MFNNS Figure 9. It does not include the preceding FP subtraction.
  */
object FPMA_Square_FP16_Golden {

  val InputWidth: Int = 16
  val MagnitudeWidth: Int = 15

  private val InputMask = 0xffff
  private val MagnitudeMask = 0x7fff
  private val MantissaIndexMask = 0x0f
  private val MantissaIndexShift = 6

  // RTL concatenation: 5'b10001 ## 2'b00 ## compensation[7:0].
  private val NegativeBiasPrefix = 0x4400

  final case class Trace(
      input: Int,
      inputSign: Int,
      magnitude: Int,
      mantissaMsb: Int,
      compensation: Int,
      correction: Int,
      shiftedMagnitude16: Int,
      shiftedMagnitude15: Int,
      resultMagnitude: Int,
      result: Int
  )

  /** Fast result-only path for exhaustive simulation sweeps. */
  def calculate(inputBits: Int): Int = {
    requireFP16Bits(inputBits)

    val magnitude = inputBits & MagnitudeMask
    val mantissaMsb = (inputBits >>> MantissaIndexShift) & MantissaIndexMask
    val compensation = CompensationTableGolden.squareComp(mantissaMsb)
    val correction = NegativeBiasPrefix | compensation
    val shiftedMagnitude15 = (magnitude << 1) & MagnitudeMask

    (shiftedMagnitude15 + correction) & MagnitudeMask
  }

  def evaluate(inputBits: Int): Trace = {
    requireFP16Bits(inputBits)

    val inputSign = (inputBits >>> 15) & 0x1
    val magnitude = inputBits & MagnitudeMask
    val mantissaMsb = (inputBits >>> MantissaIndexShift) & MantissaIndexMask
    val compensation = CompensationTableGolden.squareComp(mantissaMsb)
    val correction = NegativeBiasPrefix | compensation

    // The RTL first forms a 16-bit left shift, then feeds only bits [14:0]
    // into a 15-bit UInt adder. Carry-out from that adder is discarded.
    val shiftedMagnitude16 = (magnitude << 1) & InputMask
    val shiftedMagnitude15 = shiftedMagnitude16 & MagnitudeMask
    val resultMagnitude = (shiftedMagnitude15 + correction) & MagnitudeMask

    // A square is always emitted with a zero sign bit.
    val result = resultMagnitude

    Trace(
      input = inputBits,
      inputSign = inputSign,
      magnitude = magnitude,
      mantissaMsb = mantissaMsb,
      compensation = compensation,
      correction = correction,
      shiftedMagnitude16 = shiftedMagnitude16,
      shiftedMagnitude15 = shiftedMagnitude15,
      resultMagnitude = resultMagnitude,
      result = result
    )
  }

  private def requireFP16Bits(bits: Int): Unit = {
    require(bits >= 0 && bits <= InputMask, f"Input must be an unsigned 16-bit value: 0x$bits%x")
  }
}
