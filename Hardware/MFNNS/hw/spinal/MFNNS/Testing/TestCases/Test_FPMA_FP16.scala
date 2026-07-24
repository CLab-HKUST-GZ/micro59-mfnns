package MFNNS.Testing.TestCases

import spinal.core.sim._
import MFNNS.Config
import MFNNS.ConventionalFPMA.FPMA_FP16
import MFNNS.Testing.GoldenModels.FPMA_FP16_Golden
import scala.language.reflectiveCalls
import scala.util.Random

object Test_FPMA_FP16 {

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
  private val ExpectedRepresentativeCount = SignCount * MantissaBucketCount * SamplesPerStratum
  private val PrintSeparatorWidth = 175
  private val FinalSummaryWidth = 104
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
    val stats = new TU.TestStats("FPMA_FP16 normal-square bit-exact test")

    Config.sim
      .compile(FPMA_FP16())
      .doSim("Test_FPMA_FP16") { dut =>
        for {
          sign <- 0 until SignCount
          exponent <- NormalExponentMin to NormalExponentMax
          mantissa <- 0 until MantissaCount
        } {
          val inputBits = (sign << 15) | (exponent << 10) | mantissa

          dut.io.FP16_A #= inputBits
          dut.io.FP16_W #= inputBits
          sleep(1)

          val actualBits = dut.io.FP16_R.toInt
          val expectedBits = FPMA_FP16_Golden.calculate(inputBits, inputBits)

          stats.check(
            actual = actualBits,
            expected = expectedBits,
            clue = s"x=${TU.hex16(inputBits)}, sign=$sign, exponent=$exponent, mantissa=$mantissa"
          )

          representativeIndexByInput.get(inputBits).foreach { sampleIndex =>
            representativeResults(sampleIndex) = (actualBits, expectedBits)
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
          val (actualBits, expectedBits) = representativeResults(sampleIndex)
          printComparison(sampleIndex + 1, inputBits, actualBits, expectedBits)
        }

        println("-" * PrintSeparatorWidth)
        println(
          s"[Random Representative Output Complete] Displayed $ExpectedRepresentativeCount seeded-random rows; " +
            s"the exhaustive checker compared all ${stats.total} normal-square cases."
        )
        println()
        if (!stats.isPassed) stats.finish()
        printFinalSummary(stats)
        simSuccess()
      }
  }

  private def printHeader(): Unit = {
    println()
    println("=" * 72)
    println("[Test] FPMA_FP16 Normal-Square Bit-Exact Test")
    println("- Input domain : all normal FP16 encodings")
    println("- Operation    : FP16_A = FP16_W = x")
    println(s"- Test cases   : $ExpectedCaseCount")
    println(s"- Display only : $ExpectedRepresentativeCount seeded-random representative rows")
    println("- Reference    : FPMA_FP16_Golden")
    println("=" * 72)
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
    println("[Random Representative Output - selected rows for visual inspection]")
    println("- Sampling    : seeded stratified random sampling, without replacement")
    println(s"- Strata      : $SignCount signs x $MantissaBucketCount compensation indices")
    println(s"- Per stratum : $SamplesPerStratum randomly selected normal inputs")
    println(s"- Displayed   : $ExpectedRepresentativeCount representative rows")
    println(f"- Random seed : 0x$RepresentativeSeed%X")
    println(
      f"- Input range : E=$RepresentativeExponentMin%02d..$RepresentativeExponentMax%02d " +
        "(square results remain finite normal FP16 values)"
    )
    println("- Ordering    : grouped by sign and compensation index for readability")
    println(s"- Fully tested: all $ExpectedCaseCount normal-square inputs are checked against the Golden Model")
    println(
      "- Failure rule: every mismatch is counted; TestStats prints the first mismatching cases even if not selected"
    )
  }

  private def printRepresentativeGroupHeader(sign: Int): Unit = {
    val groupName = if (sign == 0) "positive normal inputs: (+x) * (+x)" else "negative normal inputs: (-x) * (-x)"
    println()
    println(s"[Random Representative Group] $groupName")
    println("-" * PrintSeparatorWidth)
  }

  private def printComparison(
      sampleNumber: Int,
      inputBits: Int,
      actualBits: Int,
      expectedBits: Int
  ): Unit = {
    val exponent = (inputBits >>> 10) & 0x1f
    val mantissa = inputBits & 0x3ff
    val tableIndex = mantissa / MantissaBucketWidth
    val signLabel = if ((inputBits & 0x8000) == 0) "+" else "-"
    val compensation = FPMA_FP16_Golden.evaluate(inputBits, inputBits).compensation
    val inputValue = formatValue(TU.fp16ToDouble(inputBits))
    val actualValue = formatValue(TU.fp16ToDouble(actualBits))
    val expectedValue = formatValue(TU.fp16ToDouble(expectedBits))

    val sampleColumn = "[RANDOM %02d/%02d]".format(
      sampleNumber,
      ExpectedRepresentativeCount
    )
    val inputColumn =
      f"[Input S=$signLabel E=$exponent%02d M=0x$mantissa%03x idx=$tableIndex%02d C=$compensation%03d] " +
        s"x=${TU.hex16(inputBits)} ($inputValue)"
    val dutColumn = s"[DUT] R=${TU.hex16(actualBits)} ($actualValue)"
    val goldenColumn = s"[Golden] R=${TU.hex16(expectedBits)} ($expectedValue)"
    val checkColumn =
      if (actualBits == expectedBits) "\u001B[32m\u2713 PASS\u001B[0m"
      else "\u001B[31m\u2717 FAIL\u001B[0m"

    printf(
      "%-16s | %-64s | %-33s | %-36s | [Check] %s\n",
      sampleColumn,
      inputColumn,
      dutColumn,
      goldenColumn,
      checkColumn
    )
  }

  private def printFinalSummary(stats: TU.TestStats): Unit = {
    val total = formatCount(stats.total)
    val expected = formatCount(ExpectedCaseCount)
    val passed = formatCount(stats.passed)
    val failures = formatCount(stats.failures)
    val passRate = stats.passed.toDouble * 100.0 / stats.total
    val check = s"$Green\u2713$ResetColor"

    println("=" * FinalSummaryWidth)
    println("[FINAL RESULT] FPMA_FP16 Exhaustive Normal-Square Bit-Exact Verification")
    println("-" * FinalSummaryWidth)
    println("  Operation          : FP16_A = FP16_W = x; verify FP16_R for x * x")
    println("  Verification scope : ALL normal FP16 encodings (not only the 64 displayed samples)")
    println("  Coverage formula   : 2 signs x 30 exponents (E=01..30) x 1,024 mantissas")
    println(s"  Total RTL cases    : $expected")
    println("  Golden reference   : FPMA_FP16_Golden")
    println("  Pass criterion     : DUT FP16_R[15:0] equals Golden FP16 bits exactly")
    println("-" * FinalSummaryWidth)
    println(s"  $check Exhaustive cases executed : $total / $expected")
    println(s"  $check Bit-exact Golden matches  : $passed / $expected")
    println(s"  $check DUT/Golden mismatches     : $failures")
    println(f"  $check Final pass rate           : $passRate%.3f%%")
    println("-" * FinalSummaryWidth)
    println(s"  $BoldGreen\u2713 PASS: ALL $expected NORMAL-SQUARE CASES MATCHED THE GOLDEN MODEL$ResetColor")
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
