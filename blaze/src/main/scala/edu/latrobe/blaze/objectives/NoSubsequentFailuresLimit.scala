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
 * Looks for improvement failures and flags if the number of subsequent failures
 * exceeds target. Do not use with stochastic optimizers.
 */
final class NoSubsequentFailuresLimit(override val builder: NoSubsequentFailuresLimitBuilder,
                                      override val seed:    InstanceSeed)
  extends BinaryTriggerObjective[NoSubsequentFailuresLimitBuilder](
    ObjectiveEvaluationResult.Failure
  ) {
  require(builder != null && seed != null)

  val validRange
  : RealRange = builder.validRange

  val threshold
  : Long = builder.threshold

  private var prevValue
  : Real = Real.nan

  private var noFailures
  : Long = 0L

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Boolean = {
    if (!Real.isNaN(prevValue)) {
      val diff = value - prevValue
      if (validRange.contains(diff)) {
        noFailures = 0L

      }
      else {
        noFailures += 1L
      }
    }
    prevValue = value

    noFailures >= threshold
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : ObjectiveState = NoSubsequentFailuresLimitState(
    super.state, prevValue, noFailures
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: NoSubsequentFailuresLimitState =>
        prevValue  = state.prevIntermediateValue
        noFailures = state.noFailures
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class NoSubsequentFailuresLimitBuilder()
  extends BinaryTriggerObjectiveBuilder[NoSubsequentFailuresLimitBuilder] {

  override def repr
  : NoSubsequentFailuresLimitBuilder = this

  override protected def doToString()
  : List[Any] = _validRange :: _threshold :: super.doToString()

  /**
    * Range that donats a valid outcome. Values outside are reagarded
    * as failure.
    */
  private var _validRange
  : RealRange = RealRange.zeroToInfinity

  def validRange
  : RealRange = _validRange

  def validRange_=(value: RealRange)
  : Unit = {
    require(value != null)
    _validRange = value
  }

  def setValidRange(value: RealRange)
  : NoSubsequentFailuresLimitBuilder = {
    validRange_=(value)
    this
  }

  /**
    * Number of subsequent failures permissible. Give up if
    * minimize function fails N times.
    */
  private var _threshold
  : Long = 2L

  def threshold
  : Long = _threshold

  def threshold_=(value: Long)
  : Unit = {
    require(value > 0L)
    _threshold = value
  }

  def setThreshold(value: Long)
  : NoSubsequentFailuresLimitBuilder = {
    threshold_=(value)
    this
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _validRange.hashCode())
    tmp = MurmurHash3.mix(tmp, _threshold.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[NoSubsequentFailuresLimitBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: NoSubsequentFailuresLimitBuilder =>
      _validRange == other._validRange &&
      _threshold  == other._threshold
    case _ =>
      false
  })

  override protected def doCopy()
  : NoSubsequentFailuresLimitBuilder = NoSubsequentFailuresLimitBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: NoSubsequentFailuresLimitBuilder =>
        other._validRange = _validRange
        other._threshold  = _threshold
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : NoSubsequentFailuresLimit = new NoSubsequentFailuresLimit(this, seed)

}

object NoSubsequentFailuresLimitBuilder {

  final def apply()
  : NoSubsequentFailuresLimitBuilder = new NoSubsequentFailuresLimitBuilder

  final def apply(validRange: RealRange)
  : NoSubsequentFailuresLimitBuilder = apply().setValidRange(validRange)

  final def apply(validRange: RealRange, threshold: Long)
  : NoSubsequentFailuresLimitBuilder = apply(validRange).setThreshold(threshold)

}

final case class NoSubsequentFailuresLimitState(override val parent:   InstanceState,
                                                prevIntermediateValue: Real,
                                                noFailures:            Long)
  extends ObjectiveState {
}
