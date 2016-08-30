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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * Combination of the above the bold driver with ideas from Hinton's
  * coursera course.
  */
final class HintonBoldDriver(override val builder: HintonBoldDriverBuilder,
                             override val name:    String,
                             override val seed:    InstanceSeed)
  extends IndependentParameter[HintonBoldDriverBuilder] {

  val value0
  : Real = builder.value0

  val successRange
  : RealRange = builder.successRange

  val rewardIncrement
  : Real = builder.rewardIncrement

  val penaltyFactor
  : Real = builder.penaltyFactor

  private var value
  : Real = value0

  private var valuePrev: Real = Real.nan

  override def get(phaseNo: Long)
  : Real = value

  override def update(phaseNo: Long, value: Real)
  : Unit = {
    if (!Real.isNaN(valuePrev)) {
      val delta = value - valuePrev
      if (successRange.contains(delta)) {
        this.value += rewardIncrement
      }
      else {
        this.value *= penaltyFactor
      }
    }
    valuePrev = value
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : HintonBoldDriverState = HintonBoldDriverState(super.state, value, valuePrev)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: HintonBoldDriverState =>
        value     = state.value
        valuePrev = state.valuePrev
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class HintonBoldDriverBuilder
  extends IndependentParameterBuilder[HintonBoldDriverBuilder] {

  override def repr
  : HintonBoldDriverBuilder = this

  var value0
  : Real = Real.one

  def setValue0(value: Real)
  : HintonBoldDriverBuilder = {
    value0_=(value)
    this
  }

  private var _successRange
  : RealRange = RealRange.zeroToInfinity

  def successRange
  : RealRange = _successRange

  def successRange_=(value: RealRange)
  : Unit = {
    require(value != null)
    _successRange = value
  }

  def setSuccessRange(value: RealRange)
  : HintonBoldDriverBuilder = {
    successRange_=(value)
    this
  }

  var rewardIncrement
  : Real = 0.05f

  def setRewardIncrement(value: Real)
  : HintonBoldDriverBuilder = {
    rewardIncrement_=(value)
    this
  }

  var penaltyFactor:
  Real = 0.95f

  def setPenaltyFactor(value: Real)
  : HintonBoldDriverBuilder = {
    penaltyFactor_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    f"$value0%.4g" :: _successRange :: f"$rewardIncrement%.4g" :: f"$penaltyFactor%.4g" :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, value0.hashCode())
    tmp = MurmurHash3.mix(tmp, _successRange.hashCode())
    tmp = MurmurHash3.mix(tmp, rewardIncrement.hashCode())
    tmp = MurmurHash3.mix(tmp, penaltyFactor.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[HintonBoldDriverBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: HintonBoldDriverBuilder =>
      value0           == other.value0          &&
      _successRange    == other._successRange   &&
      rewardIncrement  == other.rewardIncrement &&
      penaltyFactor    == other.penaltyFactor
    case _ =>
      false
  })

  override protected def doCopy()
  : HintonBoldDriverBuilder = HintonBoldDriverBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: HintonBoldDriverBuilder =>
        other.value0          = value0
        other._successRange   = _successRange
        other.rewardIncrement = rewardIncrement
        other.penaltyFactor   = penaltyFactor
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : HintonBoldDriver = new HintonBoldDriver(this, name, seed)

}

object HintonBoldDriverBuilder {

  final def apply()
  : HintonBoldDriverBuilder = new HintonBoldDriverBuilder

  final def apply(value0: Real)
  : HintonBoldDriverBuilder = apply().setValue0(value0)

  final def apply(value0: Real, successRange: RealRange)
  : HintonBoldDriverBuilder = apply(value0).setSuccessRange(successRange)

  final def apply(value0:          Real,
                  successRange:    RealRange,
                  rewardIncrement: Real,
                  penaltyFactor:   Real)
  : HintonBoldDriverBuilder = apply(
    value0,
    successRange
  ).setRewardIncrement(rewardIncrement).setPenaltyFactor(penaltyFactor)

}

final case class HintonBoldDriverState(override val parent: InstanceState,
                                       value:               Real,
                                       valuePrev:           Real)
  extends InstanceState {
}
