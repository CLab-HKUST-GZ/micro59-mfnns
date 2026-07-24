package MFNNS.Testing.TestCases

import MFNNS.Basics.Bin2FPCvt

import scala.math.abs

object TU {

  private val FP16Mask = 0xffff

  def checkResult(isCorrect: Boolean): String = {
    if (isCorrect) {
      "\u001B[32mCorrect\u001B[0m"
    } else {
      "\u001B[31mIncorrect\u001B[0m"
    }
  }

  def hex16(bits: Int): String = {
    requireFP16Bits(bits)
    f"0x$bits%04x"
  }

  def bin16(bits: Int): String = {
    requireFP16Bits(bits)
    String.format("%16s", bits.toBinaryString).replace(' ', '0')
  }

  def fp16ToDouble(bits: Int): Double = {
    Bin2FPCvt.FPAnyBinToFloat(
      FPBin = bin16(bits),
      ExpoWidth = 5,
      MantWidth = 10,
      CustomBias = None,
      WithNaNInf = true
    )
  }

  final class TestStats(testName: String, maxErrorPrint: Int = 10) {
    require(testName.nonEmpty, "Test name must not be empty")
    require(maxErrorPrint >= 0, "Maximum printed error count must not be negative")

    private var totalCount = 0L
    private var mismatchCount = 0L

    def total: Long = totalCount
    def failures: Long = mismatchCount
    def passed: Long = totalCount - mismatchCount
    def isPassed: Boolean = mismatchCount == 0

    def check(actual: Int, expected: Int): Unit = check(actual, expected, "")

    def check(actual: Int, expected: Int, clue: String): Unit = {
      requireFP16Bits(actual)
      requireFP16Bits(expected)
      totalCount += 1

      if (actual != expected) {
        mismatchCount += 1
        if (mismatchCount <= maxErrorPrint) {
          val clueText = clue
          val suffix = if (clueText.nonEmpty) s" | $clueText" else ""
          println(
            s"[Mismatch #$mismatchCount] expected=${hex16(expected)} (${bin16(expected)}), " +
              s"actual=${hex16(actual)} (${bin16(actual)})$suffix"
          )
        }
      }
    }

    def finish(): Unit = {
      if (isPassed) {
        println(s"[PASS] $testName: $passed/$totalCount cases passed")
      } else {
        val suppressed = mismatchCount - math.min(mismatchCount, maxErrorPrint.toLong)
        if (suppressed > 0) {
          println(s"[Info] $suppressed additional mismatches were not printed")
        }
        println(s"[FAIL] $testName: $mismatchCount/$totalCount cases failed")
        throw new AssertionError(s"$testName failed with $mismatchCount mismatches")
      }
    }
  }

  final class ErrorStats() {
    private var sampleCount = 0L
    private var relativeErrorSum = 0.0
    private var maximumRelativeError = 0.0
    private var maximumErrorCase = ""

    def count: Long = sampleCount
    def meanRelativeError: Double = if (sampleCount == 0) 0.0 else relativeErrorSum / sampleCount
    def maxRelativeError: Double = maximumRelativeError
    def maxErrorCase: String = maximumErrorCase

    def add(exactValue: Double, approximateValue: Double, caseName: String): Unit = {
      require(java.lang.Double.isFinite(exactValue), s"Exact value must be finite: $exactValue")
      require(java.lang.Double.isFinite(approximateValue), s"Approximate value must be finite: $approximateValue")
      require(exactValue != 0.0, "Relative error is undefined when the exact value is zero")

      val relativeError = abs(approximateValue - exactValue) / abs(exactValue)
      sampleCount += 1
      relativeErrorSum += relativeError

      if (sampleCount == 1 || relativeError > maximumRelativeError) {
        maximumRelativeError = relativeError
        maximumErrorCase = caseName
      }
    }
  }

  private def requireFP16Bits(bits: Int): Unit = {
    require(bits >= 0 && bits <= FP16Mask, f"Expected an unsigned 16-bit value, got 0x$bits%x")
  }
}
