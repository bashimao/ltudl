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
import edu.latrobe.blaze.modules.jvm._
import scala.util.hashing._

abstract class AddGaussianNoise
  extends AddNoiseLayer[AddGaussianNoiseBuilder] {

  final val mu
  : Real = builder.mu

  final val sigma
  : Real = builder.sigma

}

final class AddGaussianNoiseBuilder
  extends AddNoiseLayerBuilder[AddGaussianNoiseBuilder] {

  override def repr
  : AddGaussianNoiseBuilder = this

  override protected def doToString()
  : List[Any] = f"$mu%.4g" :: f"$sigma%.4g" :: super.doToString()

  var mu
  : Real = Real.zero

  def setMu(value: Real)
  : AddGaussianNoiseBuilder = {
    mu_=(value)
    this
  }

  var sigma
  : Real = Real.one

  def setSigma(value: Real)
  : AddGaussianNoiseBuilder = {
    sigma_=(value)
    this
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, mu.hashCode())
    tmp = MurmurHash3.mix(tmp, sigma.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AddGaussianNoiseBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AddGaussianNoiseBuilder =>
      mu    == other.mu &&
      sigma == other.sigma
    case _ =>
      false
  })

  override protected def doCopy()
  : AddGaussianNoiseBuilder = AddGaussianNoiseBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AddGaussianNoiseBuilder =>
        other.mu    = mu
        other.sigma = sigma
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = AddGaussianNoiseBuilder.outputPlatformFor(this, hints)

  override def build(hints:               BuildHints,
                     seed:                InstanceSeed,
                     weightBufferBuilder: ValueTensorBufferBuilder)
  : Module = AddGaussianNoiseBuilder.lookupAndBuild(
    this, hints, seed, weightBufferBuilder
  )

}

object AddGaussianNoiseBuilder
  extends ModuleVariantTable[AddGaussianNoiseBuilder] {

  register(2, AddGaussianNoise_JVM_Baseline_Description)

  final def apply()
  : AddGaussianNoiseBuilder = new AddGaussianNoiseBuilder

  final def apply(mu: Real, sigma: Real)
  : AddGaussianNoiseBuilder = apply().setMu(mu).setSigma(sigma)

}
