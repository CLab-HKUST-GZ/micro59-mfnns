package MFNNS.ConventionalFPMA

import spinal.core._
import spinal.core.sim._
import MFNNS.Config
import scala.language.postfixOps
import MFNNS.Basics.{AdderInt, FP2BinCvt, Bin2FPCvt}


// * Conventional FPMA + CompTable with 16x16 entry
case class FPMA_FP16() extends Component {
  val io = new Bundle {
    val FP16_A  = in  Bits(16 bits)
    val FP16_W  = in  Bits(16 bits)
    val FP16_R  = out Bits(16 bits)
  }
  noIoPrefix()

  val A_ExpoMant = io.FP16_A(14 downto 0)
  val W_ExpoMant = io.FP16_W(14 downto 0)

  val NegB = B"110001"      // 6 bits
  val Comp_Table = new Comp_M10_16x16()
  Comp_Table.io.A_idx := io.FP16_A(9 downto 6)
  Comp_Table.io.W_idx := io.FP16_W(9 downto 6)
  val Correction = NegB ## B"00" ## Comp_Table.io.Comp

  val FPMA_Stage1 = AdderInt(Width=16)
  FPMA_Stage1.io.X := B"0" ## A_ExpoMant
  FPMA_Stage1.io.Y := B"0" ## W_ExpoMant

  val FPMA_Stage2 = AdderInt(Width=16)
  FPMA_Stage2.io.X := FPMA_Stage1.io.Sum
  FPMA_Stage2.io.Y := Correction

  val R_ExpoMant = FPMA_Stage2.io.Sum(14 downto 0)
  val R_Sign = io.FP16_A(15) ^ io.FP16_W(15)

  io.FP16_R := R_Sign ## R_ExpoMant

}


object FPMA_FP16_Gen extends App {
  Config.setGenSubDir("/ConventionalFPMA")
  Config.spinal.generateVerilog(FPMA_FP16()).printRtl().mergeRTLSource()
}


object FPMA_FP16_Sim extends App {

  val FP16In_ValueSpace = List(
    1,2,3,4,5,6,7,8,9,10
  )

  var FP16In_Idx = 0

  Config.sim.compile{FPMA_FP16()}.doSim { dut =>
    // simulation process
    dut.clockDomain.forkStimulus(2)
    // simulation code
    for (clk <- 0 until 100) {
      // test case
      if (clk >= 10 && clk < FP16In_ValueSpace.length+10) {
        FP16In_Idx = clk - 10
        // FP16
        val FP16In = FP2BinCvt.FloatToFPAnyBin(FP16In_ValueSpace(FP16In_Idx), ExpoWidth=5, MantWidth=10, CustomBias=None, withNaNInf=true)
        dut.io.FP16_A #= BigInt(FP16In.replace("_", ""), 2)
        dut.io.FP16_W #= BigInt(FP16In.replace("_", ""), 2)
      } else {
        dut.io.FP16_A #= 0
        dut.io.FP16_W #= 0
      }
      dut.clockDomain.waitRisingEdge()    // sample on rising edge
    }
    sleep(50)
  }

}