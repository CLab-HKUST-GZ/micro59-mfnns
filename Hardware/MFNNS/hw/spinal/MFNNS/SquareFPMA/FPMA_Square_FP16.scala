package MFNNS.SquareFPMA

import spinal.core._
import spinal.core.sim._
import MFNNS.Config
import scala.language.postfixOps
import MFNNS.Basics.{FP2BinCvt, Bin2FPCvt}


// * Square FPMA + CompTable with 16 entry
case class FPMA_Square_FP16() extends Component {
  val io = new Bundle {
    val FP16_In         = in  Bits(16 bits)
    val FP16_Out        = out Bits(16 bits)
  }
  noIoPrefix()

  val ExpoMant = io.FP16_In(14 downto 0)
  val Mant = io.FP16_In(9 downto 0)
  val ExpoMantShifted = ExpoMant ## B"0"

  // Compensation for Square
  val CompSqr16 = new Comp_Square_M10_16()
  CompSqr16.io.X_MantMSB := Mant(9 downto 6)
  val Correction = B"10001" ## B"00" ## CompSqr16.io.Comp

  // Main Addition
  val ExpoMantOut = Correction.asUInt + ExpoMantShifted(14 downto 0).asUInt

  // Output
  io.FP16_Out := B"0" ## ExpoMantOut

}


object FPMA_Square_FP16_Gen extends App {
  Config.setGenSubDir("/SquareFPMA")
  Config.spinal.generateVerilog(FPMA_Square_FP16()).printRtl().mergeRTLSource()
}


object FPMA_Square_16_Sim extends App {

  val FP16In_ValueSpace = List(
    1,2,3,4,5,6,7,8,9,10
  )

  var FP16In_Idx = 0

  Config.sim.compile{FPMA_Square_FP16()}.doSim { dut =>
    // simulation process
    dut.clockDomain.forkStimulus(2)
    // simulation code
    for (clk <- 0 until 100) {
      // test case
      if (clk >= 10 && clk < FP16In_ValueSpace.length+10) {
        FP16In_Idx = clk - 10
        // FP16
        val FP16In = FP2BinCvt.FloatToFPAnyBin(FP16In_ValueSpace(FP16In_Idx), ExpoWidth=5, MantWidth=10, CustomBias=None, withNaNInf=true)
        dut.io.FP16_In #= BigInt(FP16In.replace("_", ""), 2)
      } else {
        dut.io.FP16_In #= 0
      }
      dut.clockDomain.waitRisingEdge()    // sample on rising edge
    }
    sleep(50)
  }

}