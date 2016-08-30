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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import scala.collection._

final class BenchmarkObjective(override val builder: BenchmarkObjectiveBuilder,
                               override val seed:    InstanceSeed)
  extends DependentObjectiveEx[BenchmarkObjectiveBuilder]
    with BenchmarkEnabled {
  require(builder != null && seed != null)

  override protected def doEvaluate(sink:                Sink,
                                    optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    doBenchmark(
      "evaluate",
      super.doEvaluate(
        sink,
        optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
        model,
        batch, output, value
      )
    )
  }

}

final class BenchmarkObjectiveBuilder
  extends DependentObjectiveExBuilder[BenchmarkObjectiveBuilder] {

  override def repr
  : BenchmarkObjectiveBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BenchmarkObjectiveBuilder]

  override protected def doCopy()
  : BenchmarkObjectiveBuilder = BenchmarkObjectiveBuilder()

  override def build(seed: InstanceSeed)
  : BenchmarkObjective = new BenchmarkObjective(this, seed)

}

object BenchmarkObjectiveBuilder {

  final def apply()
  : BenchmarkObjectiveBuilder = new BenchmarkObjectiveBuilder

  final def apply(child0: ObjectiveBuilder)
  : BenchmarkObjectiveBuilder = apply() += child0

  final def apply(child0: ObjectiveBuilder, childN: ObjectiveBuilder*)
  : BenchmarkObjectiveBuilder = apply(child0) ++= childN

  final def apply(childN: TraversableOnce[ObjectiveBuilder])
  : BenchmarkObjectiveBuilder = apply() ++= childN

}
