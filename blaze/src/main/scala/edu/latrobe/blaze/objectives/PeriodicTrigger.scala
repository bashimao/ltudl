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

final class PeriodicTrigger(override val builder: PeriodicTriggerBuilder,
                            override val seed:    InstanceSeed)
  extends BinaryTriggerObjective[PeriodicTriggerBuilder](
    ObjectiveEvaluationResult.Neutral
  ) {
  require(builder != null && seed != null)

  val period
  : TimeSpan = builder.period

  private var clock
  : Timer = Timer(period)

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Boolean = clock.resetIfElapsed(period)


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : ObjectiveState = PeriodicTriggerState(super.state, clock)

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: PeriodicTriggerState =>
        clock = state.clock.copy
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class PeriodicTriggerBuilder
  extends BinaryTriggerObjectiveBuilder[PeriodicTriggerBuilder] {

  override def repr
  : PeriodicTriggerBuilder = this

  private var _period
  : TimeSpan = TimeSpan.oneMinute

  def period
  : TimeSpan = _period

  def period_=(value: TimeSpan): Unit = {
    require(value >= TimeSpan.zero)
    _period = value
  }

  def setPeriod(value: TimeSpan)
  : PeriodicTriggerBuilder = {
    period_=(value)
    this
  }

  def setPeriod(value: Real)
  : PeriodicTriggerBuilder = setPeriod(TimeSpan(value))

  override protected def doToString()
  : List[Any] = _period :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _period.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PeriodicTriggerBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PeriodicTriggerBuilder =>
      _period == other._period
    case _ =>
      false
  })

  override protected def doCopy()
  : PeriodicTriggerBuilder = PeriodicTriggerBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PeriodicTriggerBuilder =>
        other._period = _period
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : PeriodicTrigger = new PeriodicTrigger(this, seed)

}

object PeriodicTriggerBuilder {

  final def apply()
  : PeriodicTriggerBuilder = new PeriodicTriggerBuilder

  final def apply(period: Real)
  : PeriodicTriggerBuilder = apply().setPeriod(period)

  final def apply(period: TimeSpan)
  : PeriodicTriggerBuilder = apply().setPeriod(period)

}

final case class PeriodicTriggerState(override val parent: InstanceState,
                                      clock:               Timer)
  extends ObjectiveState {
}
