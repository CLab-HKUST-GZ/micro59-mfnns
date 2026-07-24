package MFNNS.Basics

import scala.math._
import scala.math.pow

class CvtUtils {}


object FP2BinCvt {

  def FloatToFPAnyBin(
                       f          : Double,                  // Floating-Point format
                       ExpoWidth  : Int,
                       MantWidth  : Int,
                       CustomBias : Option[Int] = None,
                       withNaNInf : Boolean = true
                     ): String = {                           // Return FP's binary format

    def toBinary(n: Int, width: Int): String = {
      if (width <= 0) ""
      else {
        val binStr = n.toBinaryString
        val padLength = width - binStr.length
        val padded = if (padLength > 0) "0" * padLength + binStr else binStr
        padded.takeRight(width)
      }
    }

    val sign = if (f >= 0) 0 else 1

    if (f == 0.0)
      return s"${sign}_${"0" * ExpoWidth}_${"0" * MantWidth}"

    if (withNaNInf) {
      if (f.isNaN)
        return s"0_${"1" * ExpoWidth}_${"0" * (MantWidth - 1)}1"

      if (f.isInfinite)
        return s"${if (f > 0) 0 else 1}_${"1" * ExpoWidth}_${"0" * MantWidth}"
    } else if (f.isNaN || f.isInfinite) {
      return s"${sign}_${"1" * ExpoWidth}_${"1" * MantWidth}"
    }

    val absF = abs(f)
    val expo = floor(log(absF) / log(2)).toInt
    var mant = absF / pow(2, expo) - 1.0

    val fpBias = CustomBias.getOrElse((1 << (ExpoWidth - 1)) - 1)

    val (maxExponent, exponentMax) = if (withNaNInf)
      ((1 << ExpoWidth) - 2, (1 << ExpoWidth) - 2 - fpBias)
    else
      ((1 << ExpoWidth) - 1, (1 << ExpoWidth) - 1 - fpBias)

    val maxNormal = pow(2, exponentMax) * (2 - pow(2, -MantWidth))
    val minNormal = pow(2, 1 - fpBias)

    val (expoVal, mantVal) =
      if (absF < minNormal) {
        val scaledMant = absF / pow(2, 1 - fpBias)
        val mantBits = round(scaledMant * (1 << MantWidth)).min((1 << MantWidth) - 1)
        (0, mantBits)
      } else if (absF > maxNormal) {
        return s"${sign}_${"1" * ExpoWidth}_${"1" * MantWidth}"
      } else {
        val adjustedExpo = expo + fpBias
        val mantBits = round(mant * (1 << MantWidth)).min((1 << MantWidth) - 1)
        (adjustedExpo, mantBits)
      }

    s"${sign}_${toBinary(expoVal, ExpoWidth)}_${toBinary(mantVal.toInt, MantWidth)}"
  }

}


object Test1 {
  def main(args: Array[String]): Unit = {

    val IntWidth = 4
    val HalfValueSpace = pow(2, IntWidth-1).toInt

    // val ExpoWidth = 5
    // val MantWidth = 10

    val ExpoWidth = 8
    val MantWidth = 7

    // Examples
    for (i <- -HalfValueSpace until HalfValueSpace) {
      println(s"SInt=${i} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=i, ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")
    }

    val TestValue = 15360
    println(s"${TestValue} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=TestValue, ExpoWidth=5, MantWidth=10)}")

  }
}




object Bin2FPCvt {

  def FPAnyBinToFloat(
                       FPBin      : String,                  // FP's binary format
                       ExpoWidth  : Int,
                       MantWidth  : Int,
                       CustomBias : Option[Int] = None,
                       WithNaNInf : Boolean = true
                     ): Double = {                           // Return Floating-Point format

    val TotalWidth = 1 + ExpoWidth + MantWidth
    val cleaned = FPBin.replace("_", "")
    require(cleaned.length == TotalWidth, s"Input must be a $TotalWidth-bit binary string")

    val FPBias = CustomBias.getOrElse((1 << (ExpoWidth - 1)) - 1)

    // * Extract each part
    val Sign = Integer.parseInt(cleaned.substring(0, 1), 2)
    val Expo = Integer.parseInt(cleaned.substring(1, 1 + ExpoWidth), 2)
    val Mant = Integer.parseInt(cleaned.substring(1 + ExpoWidth), 2)

    val MaxExpo = (1 << ExpoWidth) - 1

    // * Sub-Normal Value
    if (Expo == 0) {
      if (Mant == 0) 0.0
      else {
        val SignFactor = if (Sign == 0) 1.0 else -1.0
        SignFactor * pow(2, 1 - FPBias) * (Mant.toDouble / pow(2, MantWidth))
      }

    // * NaN of Inf
    } else if (Expo == MaxExpo && WithNaNInf) {
      if (Mant == 0) {
        if (Sign == 0) Double.PositiveInfinity else Double.NegativeInfinity
      } else {
        Double.NaN
      }

    // * Normal Value
    } else {
      val TrueExpo = Expo - FPBias
      val TrueMant = 1.0 + Mant.toDouble / pow(2, MantWidth)
      val SignFactor = if (Sign == 0) 1.0 else -1.0
      SignFactor * pow(2, TrueExpo) * TrueMant
    }
  }

}



object Test2 {
  def main(args: Array[String]): Unit = {

    val ExpoWidth = 5
    val MantWidth = 10

    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10011_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -16
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_1110000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -15
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_1100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -14
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_1010000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -13
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_1000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -12
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_0110000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -11
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_0100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -10
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_0010000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -9
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10010_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -8
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10001_1100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -7
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10001_1000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -6
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10001_0100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -5
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10001_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -4
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10000_1000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -3
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_10000_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -2
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="1_01111_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // -1

    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_00000_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 0
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_01111_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 1
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10000_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 2
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10000_1000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 3
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10001_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 4
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10001_0100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 5
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10001_1000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 6
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10001_1100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 7
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_0000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 8
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_0010000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 9
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_0100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 10
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_0110000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 11
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_1000000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 12
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_1010000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 13
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_1100000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 14
    println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10010_1110000000", ExpoWidth=ExpoWidth, MantWidth=MantWidth)}")    // 15

  }
}



object Test3 {
  def main(args: Array[String]): Unit = {
    val DinExpoWidth  = 2
    val DinMantWidth  = 1

    val DoutExpoWidth = 8
    val DoutMantWidth = 7

    // E2M1
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_00_0", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 0.0
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_00_1", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 0.5
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_01_0", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 1.0
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_01_1", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 1.5
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10_0", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 2.0
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_10_1", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 3.0
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_11_0", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 4.0
    // println(s"${Bin2FPCvt.FPAnyBinToFloat(FPBin="0_11_1", ExpoWidth=DinExpoWidth, MantWidth=DinMantWidth, WithNaNInf=false)}")    // 6.0

    // E5M10
    println(s"${0.0} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=0.0, ExpoWidth=5, MantWidth=10)}")
    println(s"${0.5} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=0.5, ExpoWidth=5, MantWidth=10)}")
    println(s"${1.0} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=1.0, ExpoWidth=5, MantWidth=10)}")
    println(s"${1.5} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=1.5, ExpoWidth=5, MantWidth=10)}")
    println(s"${2.0} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=2.0, ExpoWidth=5, MantWidth=10)}")
    println(s"${3.0} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=3.0, ExpoWidth=5, MantWidth=10)}")
    println(s"${4.0} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=4.0, ExpoWidth=5, MantWidth=10)}")
    println(s"${6.0} -> FP=${FP2BinCvt.FloatToFPAnyBin(f=6.0, ExpoWidth=5, MantWidth=10)}")

  }
}