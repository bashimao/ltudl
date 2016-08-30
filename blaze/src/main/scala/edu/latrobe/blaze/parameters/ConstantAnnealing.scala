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
 * Implements a constant annealing schedule as described on:
 * <a href="http://www.willamette.edu/~gorr/classes/cs449/momrate.html">Link</a>
 */
final class ConstantAnnealing(override val builder: ConstantAnnealingBuilder,
                              override val name:    String,
                              override val seed:    InstanceSeed)
  extends IndependentParameter[ConstantAnnealingBuilder] {

  val value0
  : Real = builder.value0

  val rate
  : Real = builder.rate

  override def get(phaseNo: Long)
  : Real = value0 + rate * phaseNo

  override def update(phaseNo: Long, value:  Real)
  : Unit = {}

}

final class ConstantAnnealingBuilder
  extends IndependentParameterBuilder[ConstantAnnealingBuilder] {

  override def repr
  : ConstantAnnealingBuilder = this

  var value0
  : Real = Real.one

  def setValue0(value: Real)
  : ConstantAnnealingBuilder = {
    value0_=(value)
    this
  }

  var rate
  : Real = -0.0005f

  def setRate(value: Real)
  : ConstantAnnealingBuilder = {
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
  : Boolean = that.isInstanceOf[ConstantAnnealingBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConstantAnnealingBuilder =>
      value0 == other.value0 &&
      rate   == other.rate
    case _ =>
      false
  })

  override protected def doCopy()
  : ConstantAnnealingBuilder = ConstantAnnealingBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConstantAnnealingBuilder =>
        other.value0 = value0
        other.rate   = rate
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : ConstantAnnealing = new ConstantAnnealing(this, name, seed)

}

object ConstantAnnealingBuilder {

  final def apply()
  : ConstantAnnealingBuilder = new ConstantAnnealingBuilder

  final def apply(value0: Real)
  : ConstantAnnealingBuilder = apply().setValue0(value0)

  final def apply(value0: Real, rate: Real)
  : ConstantAnnealingBuilder = apply(value0).setRate(rate)

}
