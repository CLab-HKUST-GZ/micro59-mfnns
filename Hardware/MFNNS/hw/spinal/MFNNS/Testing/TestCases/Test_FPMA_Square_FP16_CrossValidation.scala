package MFNNS.Testing.TestCases

import spinal.core._
import spinal.core.sim._
import MFNNS.Config
import MFNNS.ConventionalFPMA.FPMA_FP16
import MFNNS.SquareFPMA.FPMA_Square_FP16
import scala.language.reflectiveCalls
import scala.util.Random


case class FPMA_Square_FP16_CrossDut() extends Component {
  val io = new Bundle {
    val FP16_In = in Bits (16 bits)
    val FPMA_Out = out Bits (16 bits)
    val Square_Out = out Bits (16 bits)
  }
  noIoPrefix()

  val fpma = FPMA_FP16()
  fpma.io.FP16_A := io.FP16_In
  fpma.io.FP16_W := io.FP16_In

  val squareFpma = FPMA_Square_FP16()
  squareFpma.io.FP16_In := io.FP16_In

  io.FPMA_Out := fpma.io.FP16_R
  io.Square_Out := squareFpma.io.FP16_Out
}

object Test_FPMA_Square_FP16_CrossValidation {

  private val SignCount = 2
  private val NormalExponentMin = 1
  private val NormalExponentMax = 30
  private val RepresentativeExponentMin = 8
  private val RepresentativeExponentMax = 22
  private val MantissaCount = 1024
  private val MantissaBucketWidth = 64
  private val MantissaBucketCount = MantissaCount / MantissaBucketWidth
  private val SamplesPerStratum = 2
  private val RepresentativeSeed = 0x4d464e4e53L
  private val ExpectedCaseCount =
    SignCount.toLong * (NormalExponentMax - NormalExponentMin + 1) * MantissaCount
  private val ExpectedDutEvaluationCount = ExpectedCaseCount * 2
  private val ExpectedRepresentativeCount = SignCount * MantissaBucketCount * SamplesPerStratum
  private val HeaderWidth = 104
  private val PrintSeparatorWidth = 187
  private val FinalSummaryWidth = 112
  private val Green = "\u001B[32m"
  private val BoldGreen = "\u001B[1;32m"
  private val ResetColor = "\u001B[0m"

  def main(args: Array[String]): Unit = runAll()

  def runAll(): Unit = {
    printHeader()

    val representativeInputs = selectRepresentativeInputs()
    require(
      representativeInputs.distinct.size == ExpectedRepresentativeCount,
      "Random representative inputs must be unique"
    )

    val representativeIndexByInput = representativeInputs.zipWithIndex.toMap
    val representativeResults = Array.fill[(Int, Int)](ExpectedRepresentativeCount)((0, 0))
    val representativeSeen = Array.fill(ExpectedRepresentativeCount)(false)
    val stats = new TU.TestStats("FPMA_Square_FP16 vs FPMA_FP16(x,x) RTL cross-validation")

    SimConfig
      .withConfig(Config.spinal)
      .compile(FPMA_Square_FP16_CrossDut())
      .doSim("Test_FPMA_Square_FP16_CrossValidation") { dut =>
        for {
          sign <- 0 until SignCount
          exponent <- NormalExponentMin to NormalExponentMax
          mantissa <- 0 until MantissaCount
        } {
          val inputBits = (sign << 15) | (exponent << 10) | mantissa

          dut.io.FP16_In #= inputBits
          sleep(1)

          val fpmaBits = dut.io.FPMA_Out.toInt
          val squareBits = dut.io.Square_Out.toInt

          stats.check(
            actual = squareBits,
            expected = fpmaBits,
            clue = s"x=${TU.hex16(inputBits)}, FPMA(x,x)=${TU.hex16(fpmaBits)}, " +
              s"Square-FPMA=${TU.hex16(squareBits)}, sign=$sign, exponent=$exponent, mantissa=$mantissa"
          )

          representativeIndexByInput.get(inputBits).foreach { sampleIndex =>
            representativeResults(sampleIndex) = (fpmaBits, squareBits)
            representativeSeen(sampleIndex) = true
          }
        }

        require(stats.total == ExpectedCaseCount, s"Expected $ExpectedCaseCount cases, ran ${stats.total}")
        require(
          representativeSeen.forall(identity),
          s"Expected all $ExpectedRepresentativeCount random representative inputs to be observed"
        )

        printRepresentativeHeader()
        representativeInputs.zipWithIndex.foreach { case (inputBits, sampleIndex) =>
          if (sampleIndex % (MantissaBucketCount * SamplesPerStratum) == 0) {
            printRepresentativeGroupHeader((inputBits >>> 15) & 1)
          }
          val (fpmaBits, squareBits) = representativeResults(sampleIndex)
          printComparison(sampleIndex + 1, inputBits, fpmaBits, squareBits)
        }

        println("-" * PrintSeparatorWidth)
        println(
          s"[Random Cross-Validation Output Complete] Displayed $ExpectedRepresentativeCount seeded-random rows; " +
            s"the exhaustive RTL-to-RTL checker compared all ${stats.total} paired cases."
        )
        println()
        if (!stats.isPassed) stats.finish()
        printFinalSummary(stats)
        simSuccess()
      }
  }

  private def printHeader(): Unit = {
    println()
    println("=" * HeaderWidth)
    println("[Test] FPMA_FP16 vs FPMA_Square_FP16 Exhaustive RTL-to-RTL Cross-Validation")
    println("- Baseline RTL    : FPMA_FP16 with FP16_A = FP16_W = x")
    println("- Specialized RTL : FPMA_Square_FP16 with FP16_In = x")
    println("- Input domain    : all normal FP16 encodings")
    println(s"- Paired cases    : $ExpectedCaseCount")
    println("- Comparison      : raw 16-bit output equality")
    println("- Golden model    : not used in this direct cross-check")
    println(s"- Display only    : $ExpectedRepresentativeCount seeded-random representative rows")
    println("=" * HeaderWidth)
  }

  private def selectRepresentativeInputs(): Vector[Int] = {
    val random = new Random(RepresentativeSeed)

    (for {
      sign <- 0 until SignCount
      tableIndex <- 0 until MantissaBucketCount
      inputBits <- {
        val stratum = for {
          exponent <- RepresentativeExponentMin to RepresentativeExponentMax
          lowMantissa <- 0 until MantissaBucketWidth
        } yield {
          val mantissa = tableIndex * MantissaBucketWidth + lowMantissa
          (sign << 15) | (exponent << 10) | mantissa
        }
        random.shuffle(stratum).take(SamplesPerStratum)
      }
    } yield inputBits).toVector
  }

  private def printRepresentativeHeader(): Unit = {
    println()
    println("[Random Representative RTL-to-RTL Cross-Validation Output]")
    println("- Sampling    : seeded stratified random sampling, without replacement")
    println(s"- Strata      : $SignCount signs x $MantissaBucketCount compensation indices")
    println(s"- Per stratum : $SamplesPerStratum randomly selected normal inputs")
    println(s"- Displayed   : $ExpectedRepresentativeCount representative paired comparisons")
    println(f"- Random seed : 0x$RepresentativeSeed%X")
    println(
      f"- Input range : E=$RepresentativeExponentMin%02d..$RepresentativeExponentMax%02d " +
        "(square results remain finite normal FP16 values)"
    )
    println("- Ordering    : grouped by sign and compensation index for readability")
    println(s"- Fully tested: all $ExpectedCaseCount normal inputs are compared directly between both RTL paths")
    println("- Golden model: not used in this direct RTL-to-RTL equality check")
    println(
      "- Failure rule: every mismatch is counted; TestStats prints the first mismatching cases even if not selected"
    )
  }

  private def printRepresentativeGroupHeader(sign: Int): Unit = {
    val groupName =
      if (sign == 0) "positive x: FPMA(+x,+x) vs Square-FPMA(+x)"
      else "negative x: FPMA(-x,-x) vs Square-FPMA(-x)"
    println()
    println(s"[Random Cross-Validation Group] $groupName")
    println("-" * PrintSeparatorWidth)
  }

  private def printComparison(
      sampleNumber: Int,
      inputBits: Int,
      fpmaBits: Int,
      squareBits: Int
  ): Unit = {
    val exponent = (inputBits >>> 10) & 0x1f
    val mantissa = inputBits & 0x3ff
    val tableIndex = mantissa / MantissaBucketWidth
    val signLabel = if ((inputBits & 0x8000) == 0) "+" else "-"
    val inputValue = formatValue(TU.fp16ToDouble(inputBits))
    val fpmaValue = formatValue(TU.fp16ToDouble(fpmaBits))
    val squareValue = formatValue(TU.fp16ToDouble(squareBits))

    val sampleColumn = "[CROSS %02d/%02d]".format(
      sampleNumber,
      ExpectedRepresentativeCount
    )
    val inputColumn =
      f"[Input S=$signLabel E=$exponent%02d M=0x$mantissa%03x idx=$tableIndex%02d] " +
        s"x=${TU.hex16(inputBits)} ($inputValue)"
    val fpmaColumn = s"[FPMA(x,x)] R=${TU.hex16(fpmaBits)} ($fpmaValue)"
    val squareColumn = s"[Square-FPMA] R=${TU.hex16(squareBits)} ($squareValue)"
    val checkColumn =
      if (fpmaBits == squareBits) "\u001B[32m\u2713 BIT-EXACT MATCH\u001B[0m"
      else "\u001B[31m\u2717 MISMATCH\u001B[0m"

    printf(
      "%-16s | %-58s | %-39s | %-41s | [Cross-check] %s\n",
      sampleColumn,
      inputColumn,
      fpmaColumn,
      squareColumn,
      checkColumn
    )
  }

  private def printFinalSummary(stats: TU.TestStats): Unit = {
    val total = formatCount(stats.total)
    val expected = formatCount(ExpectedCaseCount)
    val dutEvaluations = formatCount(stats.total * 2)
    val expectedDutEvaluations = formatCount(ExpectedDutEvaluationCount)
    val passed = formatCount(stats.passed)
    val failures = formatCount(stats.failures)
    val passRate = stats.passed.toDouble * 100.0 / stats.total
    val check = s"$Green\u2713$ResetColor"

    println("=" * FinalSummaryWidth)
    println("[FINAL RESULT] FPMA vs Square-FPMA Exhaustive RTL-to-RTL Cross-Validation")
    println("-" * FinalSummaryWidth)
    println("  Baseline RTL         : FPMA_FP16 with FP16_A = FP16_W = x")
    println("  Specialized RTL      : FPMA_Square_FP16 with FP16_In = x")
    println("  Input scope          : ALL normal FP16 encodings (not only the 64 displayed samples)")
    println("  Coverage formula     : 2 signs x 30 exponents (E=01..30) x 1,024 mantissas")
    println(s"  Paired comparisons   : $expected")
    println(s"  Total DUT evaluations: $expectedDutEvaluations")
    println("  Comparison basis     : direct RTL-to-RTL raw output equality; no Golden model")
    println("  Pass criterion       : Square-FPMA FP16_Out[15:0] equals FPMA FP16_R[15:0] bit-for-bit")
    println("-" * FinalSummaryWidth)
    println(s"  $check Paired cases executed    : $total / $expected")
    println(s"  $check Total DUT evaluations    : $dutEvaluations / $expectedDutEvaluations")
    println(s"  $check Bit-exact RTL matches    : $passed / $expected")
    println(s"  $check RTL-to-RTL mismatches    : $failures")
    println(f"  $check Final pass rate          : $passRate%.3f%%")
    println("-" * FinalSummaryWidth)
    println(
      s"  $BoldGreen\u2713 PASS: SQUARE-FPMA IS BIT-EXACTLY EQUIVALENT TO FPMA(x,x) " +
        s"FOR ALL $expected NORMAL INPUTS$ResetColor"
    )
    println("=" * FinalSummaryWidth)
  }

  private def formatCount(value: Long): String = f"$value%,d"

  private def formatValue(value: Double): String = {
    if (value.isNaN) "NaN"
    else if (value == Double.PositiveInfinity) "+Inf"
    else if (value == Double.NegativeInfinity) "-Inf"
    else f"$value%.9g"
  }
}
