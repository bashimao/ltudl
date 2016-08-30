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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io._
import scala.collection._
import scala.util.hashing._

/**
  * Listens to the keyboard and will tilt the value accordingly.
  */
final class Interactive(override val builder: InteractiveBuilder,
                        override val name:    String,
                        override val seed:    InstanceSeed)
  extends IndependentParameter[InteractiveBuilder] {

  private val triggers
  : Map[Int, (Real, Long, Real) => Real] = builder.triggers.toMap

  private var value
  : Real = builder.value0

  override def get(phaseNo: Long)
  : Real = value

  override def update(phaseNo: Long, value: Real)
  : Unit = {
    MapEx.foreach(triggers)((key, fn) => {
      while (LazyKeyboard.keyPressed(key)) {
        this.value = fn(this.value, phaseNo, value)
      }
    })
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : InteractiveState = InteractiveState(super.state, value)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: InteractiveState =>
        value = state.value
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class InteractiveBuilder
  extends IndependentParameterBuilder[InteractiveBuilder] {

  override def repr
  : InteractiveBuilder = this

  var value0
  : Real = Real.zero

  def setValue0(value: Real)
  : InteractiveBuilder = {
    value0_=(value)
    this
  }

  val triggers
  : mutable.Map[Int, (Real, Long, Real) => Real] = mutable.Map(
    ('+'.toInt, (vPrev, i, v) => vPrev * 1.1f),
    ('-'.toInt, (vPrev, i, v) => vPrev / 1.1f)
  )

  def +=(trigger: (Int, (Real, Long, Real) => Real))
  : InteractiveBuilder = {
    triggers += trigger
    this
  }

  def ++=(triggers: TraversableOnce[(Int, (Real, Long, Real) => Real)])
  : InteractiveBuilder = {
    this.triggers ++= triggers
    this
  }

  override protected def doToString()
  : List[Any] = f"$value0%.4g" :: triggers.size :: super.doToString()

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, value0.hashCode())
    tmp = MurmurHash3.mix(tmp, triggers.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[InteractiveBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: InteractiveBuilder =>
      value0   == other.value0 &&
      triggers == other.triggers
    case _ =>
      false
  })

  override protected def doCopy()
  : InteractiveBuilder = InteractiveBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: InteractiveBuilder =>
        other.value0 = value0
        other.triggers.clear()
        other.triggers ++= triggers
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : Interactive = new Interactive(this, name, seed)

}

object InteractiveBuilder {

  final def apply()
  : InteractiveBuilder = new InteractiveBuilder

  final def apply(value0: Real)
  : InteractiveBuilder = apply().setValue0(value0)

}

final case class InteractiveState(override val parent: InstanceState,
                                  value:               Real)
  extends InstanceState {
}
