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

package edu.latrobe.blaze.objectives.visual

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io.vega.DataSeries2D
import edu.latrobe.time.Timestamp

import scala.util.hashing.MurmurHash3

final class ParameterValueCurve(override val builder:    ParameterValueCurveBuilder,
                                override val dataSeries: DataSeries2D,
                                override val seed:       InstanceSeed)
  extends CurveEx[ParameterValueCurveBuilder] {

  val name
  : String = builder.name

  override def yValueFor(optimizer:           OptimizerLike,
                         runBeginIterationNo: Long,
                         runBeginTime:        Timestamp,
                         runNoSamples:        Long,
                         model:               Module,
                         batch:               Batch,
                         output:              Tensor,
                         value:               Real)
  : Real = {
    val iterationNo = optimizer.iterationNo
    val parameters  = optimizer.parameters
    parameters.find(
      _._2.name == name
    ).foreach(p => return p._2.get(iterationNo))
    Real.nan
  }

}

final class ParameterValueCurveBuilder
  extends CurveExBuilder[ParameterValueCurveBuilder] {

  override def repr
  : ParameterValueCurveBuilder = this

  private var _name
  : String = "lambda"

  def name
  : String = _name

  def name_=(value: String)
  : Unit = {
    require(value != null)
    _name = value
  }

  def setName(value: String)
  : ParameterValueCurveBuilder = {
    name_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _name :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _name.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ParameterValueCurveBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ParameterValueCurveBuilder =>
      _name == other._name
    case _ =>
      false
  })

  override protected def doCopy()
  : ParameterValueCurveBuilder = ParameterValueCurveBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ParameterValueCurveBuilder =>
        other._name = _name
      case _ =>
    }
  }

  override protected[visual] def doBuild(dataSeries: DataSeries2D, seed: InstanceSeed)
  : ParameterValueCurve = new ParameterValueCurve(this, dataSeries, seed)

}


object ParameterValueCurveBuilder {

  final def apply()
  : ParameterValueCurveBuilder = new ParameterValueCurveBuilder

  final def apply(name: String)
  : ParameterValueCurveBuilder = apply(name, name)

  final def apply(name: String, label: String)
  : ParameterValueCurveBuilder = apply().setName(name).setLabel(name)

}