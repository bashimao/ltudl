/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import java.io.OutputStream
import scala.collection._

/**
  * A container for objectives, that is considered successfully evaluated
  * if all sub-objectives are met.
  */
final class ComplexObjective(override val builder: ComplexObjectiveBuilder,
                             override val seed:    InstanceSeed)
  extends DependentObjective[ComplexObjectiveBuilder] {
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
    var result = Option.empty[ObjectiveEvaluationResult]
    ArrayEx.foreach(children)(child => {
      result = child.evaluate(
        sink,
        optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
        model,
        batch, output, value
      )

      // Yield execution if any child along the way failed!
      if (result.isEmpty) {
        return result
      }
    })
    result
  }

}

final class ComplexObjectiveBuilder
  extends DependentObjectiveBuilder[ComplexObjectiveBuilder] {

  override def repr
  : ComplexObjectiveBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ComplexObjectiveBuilder]

  override protected def doCopy()
  : ComplexObjectiveBuilder = ComplexObjectiveBuilder()

  override def build(seed: InstanceSeed)
  : ComplexObjective = new ComplexObjective(this, seed)

}

object ComplexObjectiveBuilder {

  final def apply()
  : ComplexObjectiveBuilder = new ComplexObjectiveBuilder

  final def apply(child0: ObjectiveBuilder)
  : ComplexObjectiveBuilder = apply() += child0

  final def apply(child0: ObjectiveBuilder, childN: ObjectiveBuilder*)
  : ComplexObjectiveBuilder = apply(child0) ++= childN

  final def apply(childN: TraversableOnce[ObjectiveBuilder])
  : ComplexObjectiveBuilder = apply() ++= childN

}
