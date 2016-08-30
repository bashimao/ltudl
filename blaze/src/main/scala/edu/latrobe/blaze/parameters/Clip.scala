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

final class Clip(override val builder: ClipBuilder,
                 override val name:    String,
                 override val seed:    InstanceSeed)
  extends DependentParameter[ClipBuilder] {

  val range
  : RealRange = builder.range

  override def get(phaseNo: Long)
  : Real = range.clip(super.get(phaseNo))

}

final class ClipBuilder
  extends DependentParameterBuilder[ClipBuilder] {

  override def repr
  : ClipBuilder = this

  private var _range
  : RealRange = RealRange.infinite

  def range
  : RealRange = _range

  def range_=(value: RealRange)
  : Unit = {
    require(value != null)
    _range = value
  }

  def setRange(value: RealRange)
  : ClipBuilder = {
    range_=(value)
    this
  }

  def setRange(min: Real, max: Real)
  : ClipBuilder = setRange(RealRange(min, max))

  override protected def doToString()
  : List[Any] = _range :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _range.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ClipBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ClipBuilder =>
      _range == other._range
    case _ =>
      false
  })

  override protected def doCopy()
  : ClipBuilder = ClipBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ClipBuilder =>
        other._range = _range
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : Clip = new Clip(this, name, seed)

}
object ClipBuilder {

  final def apply()
  : ClipBuilder = new ClipBuilder

  final def apply(source: ParameterBuilder)
  : ClipBuilder = apply().setSource(source)

  final def apply(source: ParameterBuilder, range: RealRange)
  : ClipBuilder = apply(source).setRange(range)

  final def apply(source: ParameterBuilder, min: Real, max: Real)
  : ClipBuilder = apply(source, RealRange(min, max))

}