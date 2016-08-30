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
import scala.collection._

final class MovingWindow2D
  extends DataSeries2DEx[MovingWindow2D] {

  override def repr
  : MovingWindow2D = this

  private var _noPointsMax
  : Int = 1000

  override def noPointsMax
  : Int = _noPointsMax

  def noPointsMax_=(value: Int)
  : Unit = {
    require(value > 0)
    while (points.length > value) {
      points.dequeue()
    }
    _noPointsMax = value
  }

  def setNoPointsMax(value: Int)
  : MovingWindow2D = {
    noPointsMax_=(value)
    this
  }

  override val points
  : mutable.Queue[DataPoint2D] = mutable.Queue.empty

  override def noPoints
  : Int = points.length

  override def addPoint(point: DataPoint2D)
  : Boolean = {
    if (points.length >= noPointsMax) {
      points.dequeue()
    }
    points.enqueue(point)
    true
  }

  override def replacePoint(index: Int, point: DataPoint2D)
  : Unit = points.update(index, point)

  override def clear()
  : Unit = points.clear()

  override def copy
  : MovingWindow2D = MovingWindow2D().setNoPointsMax(_noPointsMax)
}

object MovingWindow2D {

  final def apply()
  : MovingWindow2D = new MovingWindow2D

  final def apply(label: String)
  : MovingWindow2D = apply().setLabel(label)

  final def apply(label: String, color: Color)
  : MovingWindow2D = apply(label).setColor(color)

  final def apply(label: String, color: Color, noPointsMax: Int)
  : MovingWindow2D = apply(label, color).setNoPointsMax(noPointsMax)

}
