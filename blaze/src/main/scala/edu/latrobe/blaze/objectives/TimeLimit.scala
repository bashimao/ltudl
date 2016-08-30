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
 * Similar to normal relative time but can retain state over several
 * optimizer invocations.
 */
final class TimeLimit(override val builder: TimeLimitBuilder,
                      override val seed:    InstanceSeed)
  extends BinaryTriggerObjective[TimeLimitBuilder](
    ObjectiveEvaluationResult.Neutral
  ) {

  val threshold
  : TimeSpan = builder.threshold

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Boolean = {
    val beginTime = optimizer.beginTime
    TimeSpan(beginTime, Timestamp.now()) >= threshold
  }

}

final class TimeLimitBuilder
  extends BinaryTriggerObjectiveBuilder[TimeLimitBuilder] {

  override def repr
  : TimeLimitBuilder = this

  private var _threshold
  : TimeSpan = TimeSpan.oneHour

  def threshold
  : TimeSpan = _threshold

  def threshold_=(value: TimeSpan): Unit = {
    require(value >= TimeSpan.zero)
    _threshold = value
  }

  def setThreshold(value: TimeSpan)
  : TimeLimitBuilder = {
    threshold_=(value)
    this
  }

  def setThreshold(value: Real)
  : TimeLimitBuilder = setThreshold(TimeSpan(value))

  override protected def doToString()
  : List[Any] = _threshold :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _threshold.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TimeLimitBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TimeLimitBuilder =>
      _threshold == other._threshold
    case _ =>
      false
  })

  override protected def doCopy()
  : TimeLimitBuilder = TimeLimitBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: TimeLimitBuilder =>
        other._threshold = _threshold
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : TimeLimit = new TimeLimit(this, seed)

}

object TimeLimitBuilder {

  final def apply()
  : TimeLimitBuilder = new TimeLimitBuilder

  final def apply(threshold: Real)
  : TimeLimitBuilder = apply().setThreshold(threshold)

  final def apply(threshold: TimeSpan)
  : TimeLimitBuilder = apply().setThreshold(threshold)

}
