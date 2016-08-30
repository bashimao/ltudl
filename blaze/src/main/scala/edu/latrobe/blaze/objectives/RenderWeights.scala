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
import edu.latrobe.io._
import edu.latrobe.time._

/**
  * Prints the weights as JSON to the output stream. Be careful when using this.
  * This can easily overshoot the java memory allocation system of the JDK
  * which is limited to arrays with 2G elements.
  */
final class RenderWeights(override val builder: RenderWeightsBuilder,
                          override val seed:    InstanceSeed)
  extends IndependentObjective[RenderWeightsBuilder] {

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
    val clock = {
      if (logger.isDebugEnabled) {
        logger.debug("Printing model weights...")
        Stopwatch()
      }
      else {
        null
      }
    }

    // Render json into stream.
    sink.write(model.weightBuffer)

    if (clock != null) {
      logger.debug(s"Model weight printing complete. Time taken: $clock")
    }
    None
  }

}

final class RenderWeightsBuilder
  extends IndependentObjectiveBuilder[RenderWeightsBuilder] {

  override def repr
  : RenderWeightsBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RenderWeightsBuilder]

  override protected def doCopy()
  : RenderWeightsBuilder = RenderWeightsBuilder()

  override def build(seed: InstanceSeed)
  : RenderWeights = new RenderWeights(this, seed)

}

object RenderWeightsBuilder {

  final def apply()
  : RenderWeightsBuilder = new RenderWeightsBuilder

}