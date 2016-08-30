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
import edu.latrobe.time._
import edu.latrobe.io.vega._

final class IterationNoCurve(override val builder:    IterationNoCurveBuilder,
                             override val dataSeries: DataSeries2D,
                             override val seed:       InstanceSeed)
  extends CurveEx[IterationNoCurveBuilder] {

  override def yValueFor(optimizer:           OptimizerLike,
                         runBeginIterationNo: Long,
                         runBeginTime:        Timestamp,
                         runNoSamples:        Long,
                         model:               Module,
                         batch:               Batch,
                         output:              Tensor,
                         value:               Real)
  : Real = Real(optimizer.iterationNo)

}

final class IterationNoCurveBuilder
  extends CurveExBuilder[IterationNoCurveBuilder] {
  label_=("Iteration#")

  override def repr
  : IterationNoCurveBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IterationNoCurveBuilder]

  override protected def doCopy()
  : IterationNoCurveBuilder = IterationNoCurveBuilder()

  override protected[visual] def doBuild(dataSeries: DataSeries2D,
                                         seed:       InstanceSeed)
  : IterationNoCurve = new IterationNoCurve(this, dataSeries, seed)

}


object IterationNoCurveBuilder {

  final def apply()
  : IterationNoCurveBuilder = new IterationNoCurveBuilder

}
