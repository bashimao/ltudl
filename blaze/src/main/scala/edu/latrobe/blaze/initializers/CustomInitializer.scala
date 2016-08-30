/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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

import scala.util.hashing.MurmurHash3

/**
  * Simple initializers are directly backed by a distribution function from
  * which they sample values to fill the buffers.
  *
  * Use gaussian if unsure.
  *
  */
final class CustomInitializer(override val builder: CustomInitializerBuilder,
                              override val seed:    InstanceSeed)
  extends IndependentInitializer[CustomInitializerBuilder] {

  val valueFn
  : () => Real = builder.valueFn

  override def apply(module:        Module,
                     reference:     LabeledBufferReference,
                     weights:       ValueTensor,
                     inputFanSize:  Int,
                     outputFanSize: Int)
  : Unit = weights.fill(valueFn, threadSafe = false)

}

final class CustomInitializerBuilder
  extends IndependentInitializerBuilder[CustomInitializerBuilder] {

  override def repr
  : CustomInitializerBuilder = this

  private var _valueFn
  : () => Real = () => Real.zero

  def valueFn
  : () => Real = _valueFn

  def valueFn_=(value: () => Real)
  : Unit = {
    require(value != null)
    _valueFn = value
  }

  def setValueFn(value: () => Real)
  : CustomInitializerBuilder = {
    valueFn_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _valueFn :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _valueFn.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[GaussianDistributionBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CustomInitializerBuilder =>
      _valueFn == other._valueFn
    case _ =>
      false
  })

  override protected def doCopy()
  : CustomInitializerBuilder = CustomInitializerBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CustomInitializerBuilder =>
        other._valueFn = _valueFn
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : CustomInitializer = new CustomInitializer(this, seed)

}

object CustomInitializerBuilder {

  final def apply()
  : CustomInitializerBuilder = new CustomInitializerBuilder

  final def gaussian(valueFn: () => Real)
  : CustomInitializerBuilder = apply().setValueFn(valueFn)

}
