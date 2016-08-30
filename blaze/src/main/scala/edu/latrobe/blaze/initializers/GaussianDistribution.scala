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

/**
 * Random values with a gaussian distribution around mu.
 */
final class GaussianDistribution(override val builder: GaussianDistributionBuilder,
                                 override val seed:    InstanceSeed)
  extends DistributionBackedInitializer[GaussianDistributionBuilder] {

  val mu
  : Real = builder.mu

  val sigma
  : Real = builder.sigma

  override val distribution
  : Distribution[Real] = rng.gaussianDistribution(mu, sigma)

}

final class GaussianDistributionBuilder
  extends DistributionBackedInitializerBuilder[GaussianDistributionBuilder] {

  override def repr
  : GaussianDistributionBuilder = this

  var mu
  : Real = Real.zero

  def setMu(value: Real)
  : GaussianDistributionBuilder = {
    mu_=(value)
    this
  }

  var sigma
  : Real = Real.one

  def setSigma(value: Real)
  : GaussianDistributionBuilder = {
    sigma_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$mu%.4g" :: f"$sigma%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[GaussianDistributionBuilder]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, mu.hashCode())
    tmp = MurmurHash3.mix(tmp, sigma.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: GaussianDistributionBuilder =>
      mu    == other.mu &&
      sigma == other.sigma
    case _ =>
      false
  })

  override protected def doCopy()
  : GaussianDistributionBuilder = GaussianDistributionBuilder()

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: GaussianDistributionBuilder =>
        other.mu    = mu
        other.sigma = sigma
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : GaussianDistribution = new GaussianDistribution(this, seed)

}

object GaussianDistributionBuilder {

  final def apply()
  : GaussianDistributionBuilder = new GaussianDistributionBuilder

  final def apply(mu: Real, sigma: Real)
  : GaussianDistributionBuilder = apply().setMu(mu).setSigma(sigma)

}
