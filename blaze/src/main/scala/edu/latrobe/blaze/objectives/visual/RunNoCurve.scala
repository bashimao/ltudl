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

final class RunNoCurve(override val builder:    RunNoCurveCurveBuilder,
                       override val dataSeries: DataSeries2D,
                       override val seed:       InstanceSeed)
  extends CurveEx[RunNoCurveCurveBuilder] {

  override def yValueFor(optimizer:           OptimizerLike,
                         runBeginIterationNo: Long,
                         runBeginTime:        Timestamp,
                         runNoSamples:        Long,
                         model:               Module,
                         batch:               Batch,
                         output:              Tensor,
                         value:               Real)
  : Real = Real(optimizer.runNo)

}

final class RunNoCurveCurveBuilder
  extends CurveExBuilder[RunNoCurveCurveBuilder] {
  label_=("Run#")

  override def repr
  : RunNoCurveCurveBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RunNoCurveCurveBuilder]

  override protected def doCopy()
  : RunNoCurveCurveBuilder = RunNoCurveCurveBuilder()

  override protected[visual] def doBuild(dataSeries: DataSeries2D, seed: InstanceSeed)
  : RunNoCurve = new RunNoCurve(this, dataSeries, seed)

}


object RunNoCurveCurveBuilder {

  final def apply()
  : RunNoCurveCurveBuilder = new RunNoCurveCurveBuilder

}
