/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import java.io.OutputStream
import scala.util.hashing._

/**
 * If number of iterations exceeds the target count.
 */
final class RunIterationCountLimit(override val builder: RunIterationCountLimitBuilder,
                                   override val seed:    InstanceSeed)
  extends BinaryTriggerObjective[RunIterationCountLimitBuilder](
    ObjectiveEvaluationResult.Neutral
  ) {
  require(builder != null && seed != null)

  val threshold
  : Long = builder.threshold

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Boolean = {
    val runIterationNo = optimizer.iterationNo - runBeginIterationNo
    runIterationNo >= threshold
  }

}

final class RunIterationCountLimitBuilder
  extends BinaryTriggerObjectiveBuilder[RunIterationCountLimitBuilder] {

  override def repr
  : RunIterationCountLimitBuilder = this

  private var _threshold
  : Long = 100L

  def threshold
  : Long = _threshold

  def threshold_=(value: Long): Unit = {
    require(value >= 0L)
    _threshold = value
  }

  def setThreshold(value: Long)
  : RunIterationCountLimitBuilder = {
    threshold_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _threshold :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _threshold.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RunIterationCountLimitBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RunIterationCountLimitBuilder =>
      _threshold == other._threshold
    case _ =>
      false
  })

  override protected def doCopy()
  : RunIterationCountLimitBuilder = RunIterationCountLimitBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: RunIterationCountLimitBuilder =>
        other._threshold = _threshold
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : RunIterationCountLimit = new RunIterationCountLimit(this, seed)

}

object RunIterationCountLimitBuilder {

  final def apply()
  : RunIterationCountLimitBuilder = new RunIterationCountLimitBuilder

  final def apply(threshold: Long)
  : RunIterationCountLimitBuilder = apply().setThreshold(threshold)

}