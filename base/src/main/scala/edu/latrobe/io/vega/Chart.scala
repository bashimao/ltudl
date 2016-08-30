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

import java.awt.Color
import edu.latrobe._
import edu.latrobe.io._
import scala.collection._

/**
  * A vega chart.
  *
  */
// TODO: We should decouple Vega and chart rendering. See the graph rendering thingy. That is the way to go!
abstract class Chart
  extends Copyable
    with Serializable
    with JsonSerializable {

  final private var _width
  : Int = 800

  final def width
  : Int = _width

  final def width_=(value: Int)
  : Unit = {
    require(value > 0)
    _width = value
  }

  def setWidth(value: Int)
  : Chart

  final private var _height
  : Int = 400

  final def height
  : Int = _height

  final def height_=(value: Int)
  : Unit = {
    require(value > 0)
    _height = value
  }

  def setHeight(value: Int)
  : Chart

  override def copy
  : Chart

  final def copyTo(other: Chart)
  : Unit = {
    other._width  = _width
    other._height = _height
    doCopyTo(other)
  }

  protected def doCopyTo(other: Chart)
  : Unit

}

abstract class ChartEx[TThis <: ChartEx[_]]
  extends Chart
    with CopyableEx[TThis] {

  def repr
  : TThis

  final override def setWidth(value: Int): TThis = {
    width_=(value)
    repr
  }

  final override def setHeight(value: Int): TThis = {
    height_=(value)
    repr
  }

  override def copy
  : TThis = {
    val result = doCopy()
    copyTo(result)
    result
  }

  protected def doCopy()
  : TThis

}

abstract class Chart2D[TThis <: Chart2D[_]]
  extends ChartEx[TThis] {

  final val series
  : mutable.Buffer[DataSeries2D] = mutable.Buffer.empty

  final def findSeries(name: String)
  : Option[DataSeries2D] = series.find(_.label == name)

  final def getSeries(name: String)
  : DataSeries2D = findSeries(name).get

  final def getOrAddSeries(name:     String,
                           createFn: => DataSeries2D)
  : DataSeries2D = {
    val ds = findSeries(name)
    if (ds.isDefined) {
      ds.get
    }
    else {
      val ds = createFn
      series += ds
      ds
    }
  }

  /*
  def removeDataSeries(series: VegaDataSeries2D)
  : Unit = _dataSeries = _dataSeries.filter(_._2 eq series)

  final def findDataSeries(name: String)
  : Option[VegaDataSeries2D] = dataSeries.find(_._1 == name).map(_._2)

  final def getOrAddDataSeries(name:     String,
                               createFn: => VegaDataSeries2D)
  : VegaDataSeries2D = {
    val ds = findDataSeries(name)
    if (ds.isDefined) {
      ds.get
    }
    else {
      val ds = createFn
      dataSeries += name -> ds
      ds
    }
  }
  */

  final def nextColor
  : Color = {
    val ds = series
    for (c <- DefaultColors.palette) {
      if (!ds.exists(_.color == c)) {
        return c
      }
    }
    Color.BLACK
  }

}
