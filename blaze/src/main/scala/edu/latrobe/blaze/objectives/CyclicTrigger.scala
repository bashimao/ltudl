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

final class CyclicTrigger(override val builder: CyclicTriggerBuilder,
                          override val seed:    InstanceSeed)
  extends BinaryTriggerObjective[CyclicTriggerBuilder](
    ObjectiveEvaluationResult.Neutral
  ) {
  require(builder != null && seed != null)

  val cycleLength
  : Long = builder.cycleLength

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Boolean = {
    val iterationNo = optimizer.iterationNo
    iterationNo % cycleLength == 0L
  }

}

final class CyclicTriggerBuilder
  extends BinaryTriggerObjectiveBuilder[CyclicTriggerBuilder] {

  override def repr
  : CyclicTriggerBuilder = this

  private var _cycleLength
  : Long = 10L

  def cycleLength
  : Long = _cycleLength

  def cycleLength_=(value: Long)
  : Unit = {
    require(value >= 1)
    _cycleLength = value
  }

  def setCycleLength(value: Long)
  : CyclicTriggerBuilder = {
    cycleLength_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _cycleLength :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _cycleLength.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CyclicTriggerBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CyclicTriggerBuilder =>
      _cycleLength == other._cycleLength
    case _ =>
      false
  })

  override protected def doCopy()
  : CyclicTriggerBuilder = CyclicTriggerBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CyclicTriggerBuilder =>
        other._cycleLength = _cycleLength
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : CyclicTrigger = new CyclicTrigger(this, seed)

}

object CyclicTriggerBuilder {

  final def apply()
  : CyclicTriggerBuilder = new CyclicTriggerBuilder

  final def apply(cycleLength: Long)
  : CyclicTriggerBuilder = apply().setCycleLength(cycleLength)

}
