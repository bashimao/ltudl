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
import edu.latrobe.io.vega._
import edu.latrobe.time._
import java.awt.Color
import scala.util.hashing._

abstract class Curve
  extends InstanceEx[CurveBuilder] {

  def dataSeries
  : DataSeries2D

  final def update(x:                   Real,
                   optimizer:           OptimizerLike,
                   runBeginIterationNo: Long,
                   runBeginTime:        Timestamp,
                   runNoSamples:        Long,
                   model:               Module,
                   batch:               Batch,
                   output:              Tensor,
                   value:               Real)
  : Boolean = {
    val y = yValueFor(
      optimizer,
      runBeginIterationNo, runBeginTime, runNoSamples,
      model,
      batch, output, value
    )

    if (Real.isNaN(y)) {
      logger.warn("y value ignored because NaN!")
      false
    }
    else {
      dataSeries.addPoint(x, y)
    }
  }

  def yValueFor(optimizer:           OptimizerLike,
                runBeginIterationNo: Long,
                runBeginTime:        Timestamp,
                runNoSamples:        Long,
                model:               Module,
                batch:               Batch,
                output:              Tensor,
                value:               Real)
  : Real

}

abstract class CurveBuilder
  extends InstanceExBuilder2[CurveBuilder, Curve, Int, Color] {

  final private var _label
  : String = "???"

  final def label
  : String = _label

  final def label_=(value: String)
  : Unit = {
    require(value != null)
    _label = value
  }

  def setLabel(value: String)
  : CurveBuilder

  def transformLabel(fn: String => String)
  : CurveBuilder

  final private var _windowType
  : CurveWindowType = CurveWindowType.default

  final def windowType
  : CurveWindowType = _windowType

  final def windowType_=(value: CurveWindowType)
  : Unit = {
    require(value != null)
    _windowType = value
  }

  def setWindowType(value: CurveWindowType)
  : CurveBuilder

  final private var _preferredColor
  : Option[Color] = None

  final def preferredColor
  : Option[Color] = _preferredColor

  final def preferredColor_=(value: Option[Color])
  : Unit = {
    require(value != null)
    _preferredColor = value
  }

  def setPreferredColor(value: Option[Color])
  : CurveBuilder

  def setPreferredColor(value: Color)
  : CurveBuilder

  override protected def doToString()
  : List[Any] = _label :: _windowType :: _preferredColor :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _label.hashCode())
    tmp = MurmurHash3.mix(tmp, _windowType.hashCode())
    tmp = MurmurHash3.mix(tmp, _preferredColor.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CurveBuilder =>
      _label          == other._label      &&
      _windowType     == other._windowType &&
      _preferredColor == other._preferredColor
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CurveBuilder =>
        other._label          = _label
        other._windowType     = _windowType
        other._preferredColor = _preferredColor
      case _ =>
    }
  }

  final override def build(yAxisNo: Int, color: Color, seed: InstanceSeed)
  : Curve = {
    val dataSeries = _windowType.createDataSeries()
    dataSeries.setLabel(_label)
    dataSeries.setColor(_preferredColor.getOrElse(color))
    dataSeries.setYAxisNo(yAxisNo)
    doBuild(dataSeries, seed)
  }


  protected[visual] def doBuild(dataSeries: DataSeries2D, seed: InstanceSeed)
  : Curve

}

abstract class CurveEx[TBuilder <: CurveExBuilder[_]]
  extends Curve {
}

abstract class CurveExBuilder[TThis <: CurveExBuilder[_]]
  extends CurveBuilder {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

  final override def setLabel(value: String)
  : TThis = {
    label_=(value)
    repr
  }

  final override def transformLabel(fn: String => String)
  : TThis = {
    label_=(fn(label))
    repr
  }

  final override def setWindowType(value: CurveWindowType)
  : TThis = {
    windowType_=(value)
    repr
  }

  final override def setPreferredColor(value: Option[Color])
  : TThis = {
    preferredColor_=(value)
    repr
  }

  final override def setPreferredColor(value: Color)
  : TThis = setPreferredColor(Option(value))

  override protected[visual] def doBuild(dataSeries: DataSeries2D,
                                         seed:       InstanceSeed)
  : CurveEx[TThis]

}
