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

import java.awt._
import scala.collection._

final class FixedSizeWindow2D(override val noPointsMax: Int)
  extends DataSeries2DEx[FixedSizeWindow2D] {

  override def repr
  : FixedSizeWindow2D = this

  override val points
  : mutable.ArraySeq[DataPoint2D] = {
    new mutable.ArraySeq[DataPoint2D](noPointsMax)
  }

  private var _noPoints
  : Int = 0

  override def noPoints
  : Int = _noPoints

  override def addPoint(point: DataPoint2D)
  : Boolean = {
    require(_noPoints < noPointsMax)
    points(_noPoints) = point
    _noPoints += 1
    true
  }

  override def replacePoint(index: Int, point: DataPoint2D)
  : Unit = {
    require(index < _noPoints)
    points(index) = point
  }

  override def clear()
  : Unit = _noPoints = 0

  override def copy
  : FixedSizeWindow2D = FixedSizeWindow2D(noPointsMax)

}

object FixedSizeWindow2D {

  final def apply(noPointsMax: Int)
  : FixedSizeWindow2D = new FixedSizeWindow2D(noPointsMax)

  final def apply(noPointsMax: Int, label: String)
  : FixedSizeWindow2D = apply(noPointsMax).setLabel(label)

  final def apply(noPointsMax: Int, label: String, color: Color)
  : FixedSizeWindow2D = apply(noPointsMax, label).setColor(color)

}
