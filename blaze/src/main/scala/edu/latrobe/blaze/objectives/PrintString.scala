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

/**
  * As one would expect, this simply prints the specified string.
  */
final class PrintString(override val builder: PrintStringBuilder,
                        override val seed:    InstanceSeed)
  extends Print[PrintStringBuilder] {

  val value
  : String = builder.value

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : String = this.value

}

final class PrintStringBuilder
  extends PrintBuilder[PrintStringBuilder] {

  override def repr
  : PrintStringBuilder = this

  private var _value
  : String = ""

  def value
  : String = _value

  def value_=(value: String)
  : Unit = {
    require(value != null)
    _value = value
  }

  def setValue(value: String)
  : PrintStringBuilder = {
    value_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _value :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PrintStringBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _value.hashCode)

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PrintStringBuilder =>
      _value == other._value
    case _ =>
      false
  })

  override protected def doCopy()
  : PrintStringBuilder = PrintStringBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PrintStringBuilder =>
        other._value = _value
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : Objective = new PrintString(this, seed)

}

object PrintStringBuilder {

  final def apply()
  : PrintStringBuilder = new PrintStringBuilder

  final def apply(value: String)
  : PrintStringBuilder = apply().setValue(value)

  final def lineSeparator
  : PrintStringBuilder = apply(System.lineSeparator())

}
