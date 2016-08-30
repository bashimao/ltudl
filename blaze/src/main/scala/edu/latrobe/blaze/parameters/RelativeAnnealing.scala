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

final class RelativeAnnealing(override val builder: RelativeAnnealingBuilder,
                              override val name:    String,
                              override val seed:    InstanceSeed)
  extends IndependentParameter[RelativeAnnealingBuilder] {

  val value0
  : Double = DoubleEx(builder.value0)

  val rate
  : Double = DoubleEx(builder.rate)

  override def get(phaseNo: Long)
  : Real = Real(value0 * Math.pow(rate, phaseNo))

  override def update(phaseNo: Long, value:  Real)
  : Unit = {}

}

final class RelativeAnnealingBuilder
  extends IndependentParameterBuilder[RelativeAnnealingBuilder] {

  override def repr
  : RelativeAnnealingBuilder = this

  var value0
  : Real = Real.one

  def setValue0(value: Real)
  : RelativeAnnealingBuilder = {
    value0_=(value)
    this
  }

  var rate
  : Real = 0.995f

  def setRate(value: Real)
  : RelativeAnnealingBuilder = {
    rate_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$value0%.4g" :: f"$rate%.4g" :: super.doToString()

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, value0.hashCode())
    tmp = MurmurHash3.mix(tmp, rate.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RelativeAnnealingBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RelativeAnnealingBuilder =>
      value0 == other.value0 &&
      rate   == other.rate
    case _ =>
      false
  })

  override protected def doCopy()
  : RelativeAnnealingBuilder = RelativeAnnealingBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: RelativeAnnealingBuilder =>
        other.value0 = value0
        other.rate   = rate
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : RelativeAnnealing = new RelativeAnnealing(this, name, seed)

}

object RelativeAnnealingBuilder {

  final def apply()
  : RelativeAnnealingBuilder = new RelativeAnnealingBuilder

  final def apply(value0: Real)
  : RelativeAnnealingBuilder = apply().setValue0(value0)

  final def apply(value0: Real, rate: Real)
  : RelativeAnnealingBuilder = apply(value0).setRate(rate)

}
