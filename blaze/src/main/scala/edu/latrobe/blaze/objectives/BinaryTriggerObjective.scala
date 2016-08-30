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

/**
  * Base class for binary triggers. Triggers cannot access the sink and
  * eventually return either true or false.
  */
abstract class BinaryTriggerObjective[TBuilder <: BinaryTriggerObjectiveBuilder[_]](val trueResult: ObjectiveEvaluationResult)
extends IndependentObjective[TBuilder]{

  final override protected def doEvaluate(sink:                Sink,
                                          optimizer:           OptimizerLike,
                                          runBeginIterationNo: Long,
                                          runBeginTime:        Timestamp,
                                          runNoSamples:        Long,
                                          model:               Module,
                                          batch:               Batch,
                                          output:              Tensor,
                                          value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    val result = doEvaluate(
      optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
      model,
      batch, output, value
    )
    if (result) Some(trueResult) else None
  }

  protected def doEvaluate(optimizer:           OptimizerLike,
                           runBeginIterationNo: Long,
                           runBeginTime:        Timestamp,
                           runNoSamples:        Long,
                           model:               Module,
                           batch:               Batch,
                           output:              Tensor,
                           value:               Real)
  : Boolean

}

abstract class BinaryTriggerObjectiveBuilder[TThis <: BinaryTriggerObjectiveBuilder[_]]
  extends IndependentObjectiveBuilder[TThis] {
}
