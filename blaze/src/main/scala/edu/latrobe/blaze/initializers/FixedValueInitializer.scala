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

final class FixedValueInitializer(override val builder: FixedValueInitializerBuilder,
                                  override val seed:    InstanceSeed)
  extends IndependentInitializer[FixedValueInitializerBuilder] {

  override def apply(module:        Module,
                     reference:     LabeledBufferReference,
                     weights:       ValueTensor,
                     inputFanSize:  Int,
                     outputFanSize: Int)
  : Unit = weights := builder.value

}

final class FixedValueInitializerBuilder
  extends IndependentInitializerBuilder[FixedValueInitializerBuilder] {

  override def repr
  : FixedValueInitializerBuilder = this

  var value
  : Real = Real.one

  def setValue(value: Real)
  : FixedValueInitializerBuilder = {
    value_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$value%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), value.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[FixedValueInitializerBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: FixedValueInitializerBuilder =>
      value == other.value
    case _ =>
      false
  })


  override protected def doCopy()
  : FixedValueInitializerBuilder = FixedValueInitializerBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: FixedValueInitializerBuilder =>
        other.value = value
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : FixedValueInitializer = new FixedValueInitializer(this, seed)

}

object FixedValueInitializerBuilder {

  final def apply()
  : FixedValueInitializerBuilder = new FixedValueInitializerBuilder

  final def apply(value: Real)
  : FixedValueInitializerBuilder = apply().setValue(value)

  final def one
  : FixedValueInitializerBuilder = apply(Real.one)

  final def minusOne
  : FixedValueInitializerBuilder = apply(-Real.one)

  final def zero
  : FixedValueInitializerBuilder = apply(Real.zero)

}