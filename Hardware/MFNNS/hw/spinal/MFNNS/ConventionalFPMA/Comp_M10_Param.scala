package MFNNS.ConventionalFPMA

import spinal.core._
import spinal.core.sim._
import MFNNS.Config
import scala.language.postfixOps


class CompTableLut(table: CompTable2D, moduleName: String = "") extends Component {
  private val resolvedModuleName = if (moduleName.nonEmpty) moduleName else table.defaultModuleName
  setDefinitionName(resolvedModuleName)

  val io = new Bundle {
    val A_idx = in Bits(table.aIdxWidth bits)
    val W_idx = in Bits(table.wIdxWidth bits)
    val Comp  = out Bits(table.compWidth bits)
  }
  noIoPrefix()

  private val idx = io.A_idx ## io.W_idx

  io.Comp := B(0, table.compWidth bits)
  switch(idx) {
    for (a <- 0 until table.rows; w <- 0 until table.cols) {
      val linearIdx = (a << table.wIdxWidth) | w
      is(B(linearIdx, table.idxWidth bits)) {
        io.Comp := B(table.data(a)(w), table.compWidth bits)
      }
    }
  }
}


case class Comp_M10_16x16() extends CompTableLut(CompTables.E5_M10_16x16, "Comp_M10_16x16")
case class Comp_M10_8x8()   extends CompTableLut(CompTables.E5_M10_8x8,   "Comp_M10_8x8")
case class Comp_M10_4x4()   extends CompTableLut(CompTables.E5_M10_4x4,   "Comp_M10_4x4")
case class Comp_M10_2x2()   extends CompTableLut(CompTables.E5_M10_2x2,   "Comp_M10_2x2")

object Comp_M10 {
  def apply(size: Int): CompTableLut = {
    val table = CompTables.square(size)
    new CompTableLut(table, s"Comp_M10_${size}x${size}")
  }
}

object CompTableLutGen {
  def gen(table: CompTable2D): Unit = {
    Config.spinal
      .generateVerilog(new CompTableLut(table, table.defaultModuleName))
      .printRtl()
      .mergeRTLSource()
  }
}


object Comp_M10_16x16_Gen extends App {
  Config.setGenSubDir("/MFNNS/ConventionalFPMA/CompTable")
  Config.spinal.generateVerilog(Comp_M10_16x16()).printRtl().mergeRTLSource()
}



object Comp_M10_Gen extends App {
  Config.setGenSubDir("/MFNNS/ConventionalFPMA/CompTable")

  if (args.isEmpty) {
    CompTables.all.foreach(CompTableLutGen.gen)
  } else {
    args.foreach {
      case "16" | "16x16" => CompTableLutGen.gen(CompTables.square(16))
      case "8"  | "8x8"  => CompTableLutGen.gen(CompTables.square(8))
      case "4"  | "4x4"  => CompTableLutGen.gen(CompTables.square(4))
      case "2"  | "2x2"  => CompTableLutGen.gen(CompTables.square(2))
      case other => throw new IllegalArgumentException(s"Unsupported table selector: $other")
    }
  }
}
