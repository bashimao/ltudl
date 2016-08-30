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

package edu.latrobe.blaze.initializers

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

abstract class BoostingInitializer[TBuilder <: BoostingInitializerBuilder[_]]
  extends DependentInitializer[TBuilder] {

  final val gain
  : Real = builder.gain

  final override def apply(module:        Module,
                           reference:     LabeledBufferReference,
                           weights:       ValueTensor,
                           inputFanSize:  Int,
                           outputFanSize: Int)
  : Unit = {
    super.apply(
      module,
      reference,
      weights,
      inputFanSize,
      outputFanSize
    )
    weights *= gain * computeFanFactor(weights, inputFanSize, outputFanSize)
  }

  def computeFanFactor(weights:       ValueTensor,
                       inputFanSize:  Int,
                       outputFanSize: Int)
  : Real

}

abstract class BoostingInitializerBuilder[TThis <: BoostingInitializerBuilder[_]]
  extends DependentInitializerBuilder[TThis] {

  def defaultGain()
  : Real

  final var gain
  : Real = defaultGain()

  final def setGain(value: Real)
  : TThis = {
    gain_=(value)
    repr
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), gain.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BoostingInitializerBuilder[TThis] =>
      gain == other.gain
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: BoostingInitializerBuilder[TThis] =>
        other.gain = gain
      case _ =>
    }
  }

}
