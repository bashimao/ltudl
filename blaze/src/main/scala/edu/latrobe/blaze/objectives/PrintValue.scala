/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2016 Matthias Langer (t3l@threelights.de)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import java.io.OutputStreamWriter
import scala.util.hashing._

final class PrintValue(override val builder: PrintValueBuilder,
                       override val seed:    InstanceSeed)
  extends Print[PrintValueBuilder] {

  val formatFn
  : Real => String = builder.formatFn

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : String = formatFn(value)

}

final class PrintValueBuilder
  extends PrintBuilder[PrintValueBuilder] {

  override def repr
  : PrintValueBuilder = this

  private var _formatFn
  : Real => String = value => f"$value%.4g"

  def formatFn
  : Real => String = _formatFn

  def formatFn_=(value: Real => String)
  : Unit = {
    require(value != null)
    _formatFn = value
  }

  def setFormatFn(value: Real => String)
  : PrintValueBuilder = {
    formatFn_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _formatFn :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _formatFn.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PrintValueBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PrintValueBuilder =>
      _formatFn == other._formatFn
    case _ =>
      false
  })

  override protected def doCopy()
  : PrintValueBuilder = PrintValueBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PrintValueBuilder =>
        other._formatFn = _formatFn
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : PrintValue = new PrintValue(this, seed)

}

object PrintValueBuilder {

  final def apply()
  : PrintValueBuilder = new PrintValueBuilder

  final def apply(formatFn: Real => String)
  : PrintValueBuilder = apply().setFormatFn(formatFn)

  final def derive(format: String)
  : PrintValueBuilder = apply(format.format(_))

}
