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

import edu.latrobe.io.vega._

abstract class CurveWindowType
  extends Serializable {

  def createDataSeries()
  : DataSeries2D

}

object CurveWindowType {

  case class Compressing(noPointsMax: Int = 384)
    extends CurveWindowType {
    require(noPointsMax >= 2)

    override def createDataSeries()
    : DataSeries2D = CompressingWindow2D().setNoPointsMax(noPointsMax)

  }

  case class Fixed(noPointsMax: Int = 1024)
    extends CurveWindowType {
    require(noPointsMax > 0)

    override def createDataSeries()
    : DataSeries2D = FixedSizeWindow2D(noPointsMax)

  }

  case object Infinite
    extends CurveWindowType {

    override def createDataSeries()
    : DataSeries2D = InfiniteWindow2D()


  }

  case class Moving(noPointsMax: Int = 384)
    extends CurveWindowType {
    require(noPointsMax > 0)

    override def createDataSeries()
    : DataSeries2D = MovingWindow2D().setNoPointsMax(noPointsMax)

  }

  val default
  : CurveWindowType = Compressing()

}
