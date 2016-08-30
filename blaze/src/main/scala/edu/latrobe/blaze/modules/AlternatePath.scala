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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import scala.collection._
import scala.util.hashing._

/**
  * This applies the supplied augmenters in a cyclic fashion.
  */
final class AlternatePath(override val builder:             AlternatePathBuilder,
                          override val inputHints:          BuildHints,
                          override val seed:                InstanceSeed,
                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends PathSwitch[AlternatePathBuilder] {

  val interval
  : Int = builder.interval

  private var iterationNo
  : Long = 0L


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictEx(mode:           Mode,
                                     inPlaceAllowed: Boolean,
                                     input:          Tensor,
                                     reference:      Tensor,
                                     onEnter:        OnEnterPredict,
                                     onLeave:        OnLeavePredict)
  : Int = {
    val intervalNo = iterationNo / interval
    val childNo    = (intervalNo % children.length).toInt
    iterationNo += 1L

    childNo
  }


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : AlternatePathState = AlternatePathState(super.state, iterationNo)

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: AlternatePathState =>
        iterationNo = state.iterationNo
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class AlternatePathBuilder
  extends PathSwitchBuilder[AlternatePathBuilder] {

  override def repr
  : AlternatePathBuilder = this

  private var _interval
  : Int = 1

  def interval
  : Int = _interval

  def interval_=(value: Int)
  : Unit = {
    require(value > 0)
    _interval = value
  }

  def setInterval(value: Int)
  : AlternatePathBuilder = {
    interval_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _interval :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _interval.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AlternatePathBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AlternatePathBuilder =>
      _interval == other._interval
    case _ =>
      false
  })

  override protected def doCopy()
  : AlternatePathBuilder = AlternatePathBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AlternatePathBuilder =>
        other._interval = _interval
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : AlternatePath = new AlternatePath(this, hints, seed, weightsBuilder)

}

object AlternatePathBuilder {

  final def apply()
  : AlternatePathBuilder = new AlternatePathBuilder

  final def apply(module0: ModuleBuilder)
  : AlternatePathBuilder = apply() += module0

  final def apply(module0: ModuleBuilder,
                  modules: ModuleBuilder*)
  : AlternatePathBuilder = apply(module0) ++= modules

  final def apply(modules: TraversableOnce[ModuleBuilder])
  : AlternatePathBuilder = apply() ++= modules

  final def apply(modules: Array[ModuleBuilder])
  : AlternatePathBuilder = apply() ++= modules

}

final case class AlternatePathState(override val parent: InstanceState,
                                    iterationNo:         Long)
  extends ModuleState {
}
