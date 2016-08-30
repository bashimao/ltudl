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
import scala.util.hashing._
import scala.util.matching._

/**
  * Pseudo target that outputs the current parameter values.
  */
final class PrintParameters(override val builder: PrintParametersBuilder,
                            override val seed:    InstanceSeed)
  extends Print[PrintParametersBuilder] {

  val pattern
  : Regex = builder.pattern

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : String = {
    val iterationNo = optimizer.iterationNo
    val parameters = optimizer.parameters

    // Render parameter values.
    val builder = StringBuilder.newBuilder
    MapEx.foreachValue(parameters)(p => {
      if (pattern.findFirstMatchIn(p.name).isDefined) {
        p.render(iterationNo, builder)
        builder ++= ", "
      }
    })
    if (builder.nonEmpty) {
      builder.length = builder.length - 2
    }
    builder.result()
  }

}

final class PrintParametersBuilder
  extends PrintBuilder[PrintParametersBuilder] {

  override def repr
  : PrintParametersBuilder = this

  private var _pattern
  : Regex = new Regex(".*")

  def pattern
  : Regex = _pattern

  def pattern_=(value: Regex)
  : Unit = {
    require(value != null)
    _pattern = value
  }

  def setPattern(value: Regex)
  : PrintParametersBuilder = {
    pattern_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _pattern :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _pattern.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PrintParametersBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PrintParametersBuilder =>
      _pattern == other._pattern
    case _ =>
      false
  })

  override protected def doCopy()
  : PrintParametersBuilder = PrintParametersBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PrintParametersBuilder =>
        other._pattern = _pattern
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : PrintParameters = new PrintParameters(this, seed)

}

object PrintParametersBuilder {

  final def apply()
  : PrintParametersBuilder = new PrintParametersBuilder

  final def apply(pattern: Regex)
  : PrintParametersBuilder = apply().setPattern(pattern)

}