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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

final class Scale(override val builder: ScaleBuilder,
                  override val name:    String,
                  override val seed:    InstanceSeed)
  extends DependentParameter[ScaleBuilder] {

  val factor
  : Real = builder.factor

  override def get(phaseNo: Long)
  : Real = super.get(phaseNo) * factor

}

final class ScaleBuilder
  extends DependentParameterBuilder[ScaleBuilder] {

  override def repr
  : ScaleBuilder = this

  var factor
  : Real = Real.one

  def setFactor(value: Real)
  : ScaleBuilder = {
    factor_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$factor%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), factor.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ScaleBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ScaleBuilder =>
      factor == other.factor
    case _ =>
      false
  })

  override protected def doCopy()
  : ScaleBuilder = ScaleBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ScaleBuilder =>
        other.factor = factor
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : Scale = new Scale(this, name, seed)

}

object ScaleBuilder {

  final def apply()
  : ScaleBuilder = new ScaleBuilder

  final def apply(factor: Real)
  : ScaleBuilder = apply().setFactor(factor)

}