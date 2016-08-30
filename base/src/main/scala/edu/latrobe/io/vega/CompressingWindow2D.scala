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
import scala.collection._

final class CompressingWindow2D
  extends DataSeries2DEx[CompressingWindow2D] {

  override def repr
  : CompressingWindow2D = this

  override def noPointsMax
  : Int = _points.length

  def noPointsMax_=(value: Int)
  : Unit = {
    require(value >= 0)
    _points   = java.util.Arrays.copyOf(_points, value)
    _noPoints = Math.min(_noPoints, _points.length)
  }

  def setNoPointsMax(value: Int)
  : CompressingWindow2D = {
    noPointsMax_=(value)
    this
  }

  private var _points
  : Array[DataPoint2D] = new Array[DataPoint2D](1000)

  override def points
  : Seq[DataPoint2D] = ArrayEx.take(_points, _noPoints)

  private var _noPoints
  : Int = 0

  override def noPoints
  : Int = _noPoints

  private var pointWeightMax
  : Int = 1

  private var pointWeight
  : Int = 0

  override def addPoint(point: DataPoint2D)
  : Boolean = {
    // Compress everything by factor 2.
    if (_noPoints >= _points.length) {
      var j = 0
      var i = 0
      while (i < _noPoints - 1) {
        //val x = _points(i)._1
        //val x = MathMacros.lerp(_points(i)._1, _points(i + 1)._1, Real.pointFive)
        //val y = MathMacros.lerp(_points(i)._2, _points(i + 1)._2, Real.pointFive)
        //_points(j) = (x, y)
        _points(j) = _points(i).lerp(_points(i + 1), Real.pointFive)
        i += 2
        j += 1
      }
      if (i < _noPoints) {
        _points(j) = _points(i)
      }
      else {
        pointWeight = 0
      }
      _noPoints = j
      pointWeightMax *= 2
    }

    // Add new point.
    pointWeight += 1
    if (pointWeight == 1) {
      _points(_noPoints) = point
    }
    else {
      //val x = _points(_noPoints)._1
      //val x = MathMacros.lerp(_points(_noPoints)._1, point._1, Real.one / pointWeight)
      //val y = MathMacros.lerp(_points(_noPoints)._2, point._2, Real.one / pointWeight)
      //_points(_noPoints) = (x, y)
      _points(_noPoints) = _points(_noPoints).lerp(point, Real.one / pointWeight)
    }

    if (pointWeight >= pointWeightMax) {
      _noPoints += 1
      pointWeight = 0
      true
    }
    else {
      false
    }
  }

  override def replacePoint(index: Int, point: DataPoint2D)
  : Unit = throw new UnsupportedOperationException

  override def clear()
  : Unit = {
    _noPoints      = 0
    pointWeight    = 1
    pointWeightMax = 1
  }

  override def copy
  : CompressingWindow2D = CompressingWindow2D().setNoPointsMax(noPointsMax)

}

object CompressingWindow2D {

  final def apply()
  : CompressingWindow2D = new CompressingWindow2D

  final def apply(label: String)
  : CompressingWindow2D = apply().setLabel(label)

  final def apply(label: String, color: Color)
  : CompressingWindow2D = apply(label).setColor(color)

  final def apply(label: String, color: Color, noPointsMax: Int)
  : CompressingWindow2D = apply(label, color).setNoPointsMax(noPointsMax)

}
