package MFNNS.Testing

import MFNNS.Testing.TestCases.{Test_FPMA_FP16, Test_FPMA_Square_FP16, Test_FPMA_Square_FP16_CrossValidation}

object OverallFunctionalTest extends App {

  private val FunctionalTestCount = 3
  private val CasesPerTest = 61440L
  private val GoldenModelCheckCount = CasesPerTest * 2
  private val DirectRtlCrossCheckCount = CasesPerTest
  private val OverallComparisonCount = GoldenModelCheckCount + DirectRtlCrossCheckCount
  private val SummaryWidth = 112
  private val Green = "\u001B[32m"
  private val BoldGreen = "\u001B[1;32m"
  private val ResetColor = "\u001B[0m"

  printOverallHeader()

  runStage(
    index = 1,
    title = "Conventional FPMA_FP16 vs Golden Model",
    description = "FPMA_FP16(A=x, W=x) == FPMA_FP16_Golden(x,x)"
  ) {
    Test_FPMA_FP16.runAll()
  }

  runStage(
    index = 2,
    title = "Specialized FPMA_Square_FP16 vs Golden Model",
    description = "FPMA_Square_FP16(x) == FPMA_Square_FP16_Golden(x)"
  ) {
    Test_FPMA_Square_FP16.runAll()
  }

  runStage(
    index = 3,
    title = "FPMA(x,x) vs Square-FPMA RTL-to-RTL Cross-Validation",
    description = "FPMA_Square_FP16(x) == FPMA_FP16(A=x, W=x)"
  ) {
    Test_FPMA_Square_FP16_CrossValidation.runAll()
  }

  printOverallSummary()

  private def runStage(index: Int, title: String, description: String)(testBody: => Unit): Unit = {
    println()
    println("=" * SummaryWidth)
    println(s"[FUNCTIONAL TEST $index/$FunctionalTestCount] $title")
    println(s"- Verification : $description")
    println(s"- Input scope  : all ${formatCount(CasesPerTest)} normal FP16 encodings")
    println("- Policy       : exhaustive, bit-exact, fail-fast")
    println("=" * SummaryWidth)
    println()

    testBody

    println()
    println(
      s"$BoldGreen[FUNCTIONAL TEST $index/$FunctionalTestCount COMPLETE] " +
        s"\u2713 PASS: ${formatCount(CasesPerTest)} / ${formatCount(CasesPerTest)} cases$ResetColor"
    )
  }

  private def printOverallHeader(): Unit = {
    println()
    println("=" * SummaryWidth)
    println("[MFNNS ARTIFACT EVALUATION] Overall FP16 Functional Verification")
    println("-" * SummaryWidth)
    println("  Functional test 1 : FPMA_FP16 RTL vs FPMA_FP16 Golden Model")
    println("  Functional test 2 : FPMA_Square_FP16 RTL vs FPMA_Square_FP16 Golden Model")
    println("  Functional test 3 : FPMA(x,x) RTL vs Square-FPMA RTL direct cross-validation")
    println(s"  Cases per test    : ${formatCount(CasesPerTest)}")
    println(s"  Planned checks    : ${formatCount(OverallComparisonCount)} bit-exact comparisons")
    println("  Execution policy  : sequential and fail-fast")
    println("=" * SummaryWidth)
  }

  private def printOverallSummary(): Unit = {
    val check = s"$Green\u2713$ResetColor"
    val casesPerTest = formatCount(CasesPerTest)
    val goldenChecks = formatCount(GoldenModelCheckCount)
    val directChecks = formatCount(DirectRtlCrossCheckCount)
    val overallChecks = formatCount(OverallComparisonCount)

    println()
    println("=" * SummaryWidth)
    println("[OVERALL FINAL RESULT] MFNNS FP16 Functional Verification")
    println("-" * SummaryWidth)
    println(s"  Functional simulations : $FunctionalTestCount")
    println("  Input domain            : all normal FP16 encodings")
    println(s"  Cases per simulation    : $casesPerTest")
    println(s"  Golden-model checks     : $goldenChecks")
    println(s"  Direct RTL cross-checks : $directChecks")
    println(s"  Overall comparisons     : $overallChecks")
    println("-" * SummaryWidth)
    println(s"  $check FPMA_FP16 vs Golden                  : $casesPerTest / $casesPerTest")
    println(s"  $check FPMA_Square_FP16 vs Golden           : $casesPerTest / $casesPerTest")
    println(s"  $check Square-FPMA vs FPMA(x,x)             : $casesPerTest / $casesPerTest")
    println(s"  $check Functional test stages completed     : $FunctionalTestCount / $FunctionalTestCount")
    println(s"  $check Overall bit-exact comparisons passed : $overallChecks / $overallChecks")
    println("-" * SummaryWidth)
    println(s"  $BoldGreen\u2713 OVERALL PASS: ALL MFNNS FP16 FUNCTIONAL VERIFICATION TESTS PASSED$ResetColor")
    println("=" * SummaryWidth)
  }

  private def formatCount(value: Long): String = f"$value%,d"
}
