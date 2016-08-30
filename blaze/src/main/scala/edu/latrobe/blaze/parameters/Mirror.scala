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

/**
  * Mirrors the value at the specified constant value.
  *
  * f(x) = c - x
  *
  */
final class Mirror(override val builder: MirrorBuilder,
                   override val name:    String,
                   override val seed:    InstanceSeed)
  extends DependentParameter[MirrorBuilder] {

  val value
    : Real = builder.value

  override def get(phaseNo: Long)
  : Real = value - super.get(phaseNo)

}

final class MirrorBuilder
  extends DependentParameterBuilder[MirrorBuilder] {

  override def repr
  : MirrorBuilder = this

  var value
  : Real = Real.zero

  def setValue(value: Real)
  : MirrorBuilder = {
    value_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$value%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MirrorBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), value.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MirrorBuilder =>
      value == other.value
    case _ =>
      false
  })

  override protected def doCopy()
  : MirrorBuilder = MirrorBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: MirrorBuilder =>
        other.value = value
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : Mirror = new Mirror(this, name, seed)

}

object MirrorBuilder {

  final def apply()
  : MirrorBuilder = new MirrorBuilder

  final def apply(source: ParameterBuilder)
  : MirrorBuilder = apply().setSource(source)

  final def apply(source: ParameterBuilder, value: Real)
  : MirrorBuilder = apply(source).setValue(value)

}
