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

package edu.latrobe.blaze.initializers

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

final class BernoulliDistribution(override val builder: BernoulliDistributionBuilder,
                                  override val seed:    InstanceSeed)
  extends DistributionBackedInitializer[BernoulliDistributionBuilder] {
  require(builder != null && seed != null)

  val probability
  : Real = builder.probability

  override val distribution
  : Distribution[Real] = rng.bernoulliDistribution(
    probability,
    Real.zero,
    Real.one
  )

}

final class BernoulliDistributionBuilder
  extends DistributionBackedInitializerBuilder[BernoulliDistributionBuilder] {

  override def repr
  : BernoulliDistributionBuilder = this

  private var _probability
  : Real = Real.pointFive

  def probability
  : Real = _probability

  def probability_=(value: Real)
  : Unit = {
    require(value >= Real.zero && value <= Real.one)
    _probability = value
  }

  def setProbability(value: Real)
  : BernoulliDistributionBuilder = {
    probability_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_probability}%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BernoulliDistributionBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _probability.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BernoulliDistributionBuilder =>
      _probability == other._probability
    case _ =>
      false
  })

  override protected def doCopy()
  : BernoulliDistributionBuilder = BernoulliDistributionBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: BernoulliDistributionBuilder =>
        other._probability = _probability
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : BernoulliDistribution = new BernoulliDistribution(this, seed)

}

object BernoulliDistributionBuilder {

  final def apply()
  : BernoulliDistributionBuilder = new BernoulliDistributionBuilder

  final def apply(probability: Real)
  : BernoulliDistributionBuilder = apply().setProbability(probability)

}
