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

import breeze.stats.distributions._
import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
 * Uniformly distributed random values.
 */
final class UniformDistribution(override val builder: UniformDistributionBuilder,
                                override val seed:    InstanceSeed)
  extends DistributionBackedInitializer[UniformDistributionBuilder] {

  val range
  : RealRange = builder.range

  override val distribution
  : Distribution[Real] = rng.uniformDistribution(range)

}

final class UniformDistributionBuilder
  extends DistributionBackedInitializerBuilder[UniformDistributionBuilder] {

  override def repr
  : UniformDistributionBuilder = this

  /**
    * Because we use stddev by default. 3x stddev = 99.7%. So the default
    * uniform and the stddev cover about the same numeric range.
    */
  private var _range
  : RealRange = RealRange(-3.0f, 3.0f)

  def range
  : RealRange = _range

  def range_=(value: RealRange)
  : Unit = {
    require(value != null)
    _range = value
  }

  def setRange(value: RealRange)
  : UniformDistributionBuilder = {
    range_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _range :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[UniformDistributionBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _range.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: UniformDistributionBuilder =>
      _range == other._range
    case _ =>
      false
  })


  override protected def doCopy()
  : UniformDistributionBuilder = UniformDistributionBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: UniformDistributionBuilder =>
        other._range = _range
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : UniformDistribution = new UniformDistribution(this, seed)

}

object UniformDistributionBuilder {

  final def apply()
  : UniformDistributionBuilder = new UniformDistributionBuilder

  final def apply(range: RealRange)
  : UniformDistributionBuilder = apply().setRange(range)

  final def apply(min: Real, max: Real)
  : UniformDistributionBuilder = apply(RealRange(min, max))

}
