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
import edu.latrobe.io.vega.DataSeries2D
import edu.latrobe.time.Timestamp

final class ValueCurve(override val builder:    ValueCurveBuilder,
                       override val dataSeries: DataSeries2D,
                       override val seed:       InstanceSeed)
  extends CurveEx[ValueCurveBuilder] {

  override def yValueFor(optimizer:           OptimizerLike,
                         runBeginIterationNo: Long,
                         runBeginTime:        Timestamp,
                         runNoSamples:        Long,
                         model:               Module,
                         batch:               Batch,
                         output:              Tensor,
                         value:               Real)
  : Real = value

}

final class ValueCurveBuilder
  extends CurveExBuilder[ValueCurveBuilder] {
  label_=("f(x|w)")

  override def repr
  : ValueCurveBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ValueCurveBuilder]

  override protected def doCopy()
  : ValueCurveBuilder = ValueCurveBuilder()

  override protected[visual] def doBuild(dataSeries: DataSeries2D, seed: InstanceSeed)
  : ValueCurve = new ValueCurve(this, dataSeries, seed)

}


object ValueCurveBuilder {

  final def apply()
  : ValueCurveBuilder = new ValueCurveBuilder

}