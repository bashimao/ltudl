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

import edu.latrobe.{Equatable, Real}
import edu.latrobe.blaze._

import scala.util.hashing._

/**
 * Keeps learning rate fixed.
 */
final class ConstantValue(override val builder: ConstantValueBuilder,
                          override val name:    String,
                          override val seed:    InstanceSeed)
  extends IndependentParameter[ConstantValueBuilder] {

  val value
  : Real = builder.value

  override def get(phaseNo: Long)
  : Real = value

  override def update(phaseNo: Long, value:  Real)
  : Unit = {}

}

final class ConstantValueBuilder
  extends IndependentParameterBuilder[ConstantValueBuilder] {

  override def repr
  : ConstantValueBuilder = this

  var value
  : Real = Real.zero

  def setValue(value: Real)
  : ConstantValueBuilder = {
    value_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$value%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), value.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConstantValueBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConstantValueBuilder =>
      value == other.value
    case _ =>
      false
  })

  override protected def doCopy()
  : ConstantValueBuilder = ConstantValueBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConstantValueBuilder =>
        other.value = value
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : ConstantValue = new ConstantValue(this, name, seed)

}

object ConstantValueBuilder {

  final def apply()
  : ConstantValueBuilder = new ConstantValueBuilder

  final def apply(value: Real)
  : ConstantValueBuilder = apply().setValue(value)

  final def one
  : ConstantValueBuilder = apply(Real.one)

  final def zero
  : ConstantValueBuilder = apply(Real.zero)

  final def nan
  : ConstantValueBuilder = apply(Real.nan)

}