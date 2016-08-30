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

package edu.latrobe.io.vega

import edu.latrobe._
import java.awt.Color
import org.json4s.JsonAST._
import scala.collection._

abstract class DataSeries2D
  extends Serializable
    with JsonSerializable {

  final private var _label
  : String = "~"

  final def label
  : String = _label

  final def label_=(value: String)
  : Unit = {
    require(value != null)
    _label = value
  }

  def setLabel(value: String)
  : DataSeries2D

  final private var _color
  : Color = Color.BLACK

  final def color
  : Color = _color

  final def color_=(value: Color)
  : Unit = {
    require(value != null)
    _color = value
  }

  def setColor(value: Color)
  : DataSeries2D

  final private var _lineWidth
  : Int = 1

  final def lineWidth
  : Int = _lineWidth

  final def lineWidth_=(value: Int)
  : Unit = {
    require(value > 0)
    _lineWidth = value
  }

  def setLineWidth(value: Int)
  : DataSeries2D

  final private var _yAxisNo
  : Int = 0

  final def yAxisNo
  : Int = _yAxisNo

  final def yAxisNo_=(value: Int)
  : Unit = {
    require(value >= 0 && value <= 1)
    _yAxisNo = value
  }

  def setYAxisNo(value: Int)
  : DataSeries2D

  final private var _symbolSize
  : Int = 0

  final def symbolSize
  : Int = _symbolSize

  final def symbolSize_=(value: Int)
  : Unit = {
    require(value >= 0)
    _symbolSize = value
  }

  def setSymbolSize(value: Int)
  : DataSeries2D

  final private var _symbolShape
  : String = "circle"

  final def symbolShape
  : String = _symbolShape

  final def symbolShape_=(value: String)
  : Unit = {
    require(value != null)
    _symbolShape = value
  }

  def setSymbolShape(value: String)
  : DataSeries2D

  final var opacity
  : Int = 1

  def setOpacity(value: Int)
  : DataSeries2D

  def points
  : Seq[DataPoint2D]

  def noPoints
  : Int

  def noPointsMax
  : Int

  def isEmpty
  : Boolean = noPoints == 0

  def nonEmpty
  : Boolean = noPoints != 0

  final def addPoint(x: Real, y: Real)
  : Boolean = addPoint(DataPoint2D(x, y))

  /**
    * @return True if graph-redraw required.
    */
  def addPoint(point: DataPoint2D)
  : Boolean

  def addPoints(points: Array[DataPoint2D])
  : Unit = {
    ArrayEx.foldLeft(
      0,
      points
    )((res, p) => if (addPoint(p)) res + 1 else res)
  }

  final def replacePoint(index: Int, x: Real, y: Real)
  : Unit = replacePoint(index, DataPoint2D(x, y))

  def replacePoint(index: Int, point: DataPoint2D)
  : Unit

  final def replacePoints(fn: Int => DataPoint2D)
  : Unit = {
    val n = noPoints
    var i = 0
    while (i < n) {
      replacePoint(i, fn(i))
      i += 1
    }
  }

  def clear()
  : Unit

  final def minX
  : Real = points.minBy(_.x).x

  final def maxX
  : Real = points.maxBy(_.x).x

  final def minY
  : Real = points.maxBy(_.y).y

  final def maxY
  : Real = points.maxBy(_.y).y

  override protected def doToJson()
  : List[JField] = List(
    Json.field("name", _label),
    Json.field("values", {
      val builder = List.newBuilder[JObject]
      points.foreach(
        builder += _.toJson
      )
      builder.result()
    })
  )

}

abstract class DataSeries2DEx[TThis <: DataSeries2DEx[_]]
  extends DataSeries2D
    with CopyableEx[TThis] {

  def repr
  : TThis

  final override def setLabel(value: String)
  : TThis = {
    label_=(value)
    repr
  }

  final override def setColor(value: Color)
  : TThis = {
    color_=(value)
    repr
  }

  final override def setLineWidth(value: Int)
  : TThis = {
    lineWidth_=(value)
    repr
  }

  final override def setYAxisNo(value: Int)
  : TThis = {
    yAxisNo_=(value)
    repr
  }

  final override def setSymbolShape(value: String)
  : TThis = {
    symbolShape_=(value)
    repr
  }

  final override def setSymbolSize(value: Int)
  : TThis = {
    symbolSize_=(value)
    repr
  }

  final override def setOpacity(value: Int)
  : TThis = {
    opacity_=(value)
    repr
  }

}
