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

abstract class AddUniformNoise
  extends AddNoiseLayer[AddUniformNoiseBuilder] {

  final val range
  : RealRange = builder.range

}

final class AddUniformNoiseBuilder
  extends AddNoiseLayerBuilder[AddUniformNoiseBuilder] {

  override def repr
  : AddUniformNoiseBuilder = this

  override protected def doToString()
  : List[Any] = _range :: super.doToString()

  private var _range
  : RealRange = RealRange.minusOneToOne

  def range
  : RealRange = _range

  def range_=(value: RealRange)
  : Unit = {
    require(value != null)
    _range = value
  }

  def setRange(value: RealRange)
  : AddUniformNoiseBuilder = {
    range_=(value)
    this
  }

  override def hashCode()
  : Int =  MurmurHash3.mix(super.hashCode(), _range.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AddUniformNoiseBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AddUniformNoiseBuilder =>
      _range == other._range
    case _ =>
      false
  })

  override protected def doCopy()
  : AddUniformNoiseBuilder = AddUniformNoiseBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AddUniformNoiseBuilder =>
        other._range = _range
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = AddUniformNoiseBuilder.outputPlatformFor(this, hints)

  override def build(hints:               BuildHints,
                     seed:                InstanceSeed,
                     weightBufferBuilder: ValueTensorBufferBuilder)
  : Module = AddUniformNoiseBuilder.lookupAndBuild(
    this, hints, seed, weightBufferBuilder
  )

}

object AddUniformNoiseBuilder
  extends ModuleVariantTable[AddUniformNoiseBuilder] {

  register(2, AddUniformNoise_JVM_Baseline_Description)

  final def apply()
  : AddUniformNoiseBuilder = new AddUniformNoiseBuilder

  final def apply(range: RealRange)
  : AddUniformNoiseBuilder = apply().setRange(range)

  final def apply(min: Real, max: Real)
  : AddUniformNoiseBuilder = apply(RealRange(min, max))

}
