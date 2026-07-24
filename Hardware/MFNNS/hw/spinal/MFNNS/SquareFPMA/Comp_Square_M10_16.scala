package MFNNS.SquareFPMA

import spinal.core._
import spinal.core.sim._
import MFNNS.Config
import scala.language.postfixOps


// Down-sampling Compensation
case class Comp_Square_M10_16() extends Component {
  val io = new Bundle {
    val X_MantMSB = in  Bits(4 bits)
    val Comp      = out Bits(8 bits)
  }
  noIoPrefix()

  val InIdx = io.X_MantMSB

  switch(InIdx) {
    is(B"0000") { io.Comp := B"00000001" }    // 1
    is(B"0001") { io.Comp := B"00001001" }    // 9
    is(B"0010") { io.Comp := B"00011001" }    // 25
    is(B"0011") { io.Comp := B"00110001" }    // 49
    is(B"0100") { io.Comp := B"01010001" }    // 81
    is(B"0101") { io.Comp := B"01111001" }    // 121
    is(B"0110") { io.Comp := B"10100110" }    // 166
    is(B"0111") { io.Comp := B"10010001" }    // 145
    is(B"1000") { io.Comp := B"01110001" }    // 113
    is(B"1001") { io.Comp := B"01010101" }    // 85
    is(B"1010") { io.Comp := B"00111101" }    // 61
    is(B"1011") { io.Comp := B"00101001" }    // 41
    is(B"1100") { io.Comp := B"00011001" }    // 25
    is(B"1101") { io.Comp := B"00001101" }    // 13
    is(B"1110") { io.Comp := B"00000101" }    // 5
    is(B"1111") { io.Comp := B"00000000" }    // 0
  }

}


object Comp_Square_M10_16_Gen extends App {
  Config.setGenSubDir("/SquareFPMA")
  Config.spinal.generateVerilog(Comp_Square_M10_16()).printRtl().mergeRTLSource()
}