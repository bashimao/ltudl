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
import scala.collection._

final class InfiniteWindow2D
  extends DataSeries2DEx[InfiniteWindow2D] {

  override def repr
  : InfiniteWindow2D = this

  override val noPointsMax
  : Int = ArrayEx.maxSize

  override val points
  : mutable.Buffer[DataPoint2D] = mutable.Buffer.empty

  override def noPoints
  : Int = points.length

  override def addPoint(point: DataPoint2D)
  : Boolean = {
    require(points.length < noPointsMax)
    points += point
    true
  }

  override def replacePoint(index: Int, point: DataPoint2D)
  : Unit = {
    require(index >= points.length)
    points.update(index, point)
  }

  override def clear()
  : Unit = points.clear()

  override def copy
  : InfiniteWindow2D = InfiniteWindow2D()

}

object InfiniteWindow2D {

  final def apply()
  : InfiniteWindow2D = new InfiniteWindow2D

  final def apply(label: String)
  : InfiniteWindow2D = apply().setLabel(label)

  final def apply(label: String, color: Color)
  : InfiniteWindow2D = apply(label).setColor(color)

}
