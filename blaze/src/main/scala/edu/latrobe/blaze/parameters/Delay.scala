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
import scala.util.hashing._

final class Delay(override val builder: DelayBuilder,
                  override val name:    String,
                  override val seed:    InstanceSeed)
  extends DependentParameter[DelayBuilder] {

  val noIterations
  : Long = builder.noIterations

  override def get(phaseNo: Long)
  : Real = {
    val i = Math.max(phaseNo - noIterations, 0L)
    super.get(i)
  }

  override def update(phaseNo: Long, value: Real)
  : Unit = {
    val i = phaseNo - noIterations
    if (i >= 0L) {
      super.update(i, value)
    }
  }

}

final class DelayBuilder
  extends DependentParameterBuilder[DelayBuilder] {

  override def repr
  : DelayBuilder = this

  private var _noIterations
  : Long = 0L

  def noIterations
  : Long = _noIterations

  def noIterations_=(value: Long)
  : Unit = {
    require(value >= 0L)
    _noIterations = value
  }

  def setNoIterations(value: Long)
  : DelayBuilder = {
    noIterations_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _noIterations :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _noIterations.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DelayBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DelayBuilder =>
      _noIterations == other._noIterations
    case _ =>
      false
  })

  override protected def doCopy()
  : DelayBuilder = DelayBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DelayBuilder =>
        other._noIterations = _noIterations
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : Delay = new Delay(this, name, seed)

}

object DelayBuilder {

  final def apply()
  : DelayBuilder = new DelayBuilder

  final def apply(noIterations: Long)
  : DelayBuilder = apply().setNoIterations(noIterations)

  final def apply(noIterations: Long, source: ParameterBuilder)
  : DelayBuilder = apply(noIterations).setSource(source)

}
