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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._

trait TrainableLayer[TThis <: TrainableLayerBuilder[_]]
  extends Layer[TThis] {

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveGradients(input:     Tensor,
                                                 reference: Tensor,
                                                 output:    Tensor,
                                                 context:   PredictContext,
                                                 error:     NextError,
                                                 sink:      ValueTensorBuffer)
  : NextError = {
    val oldErr = error.compute()

    val clock = if (logger.isTraceEnabled) Stopwatch() else null

    // Compute gradients towards weights.
    doDeriveWeightGradients(
      input,
      reference,
      output,
      context,
      oldErr,
      sink
    )

    if (clock != null) {
      logger.trace(
        f"$clock%s => deriveWeightGradients(${oldErr.platform}%-4s) => $this%s"
      )
    }

    // Prepare computation of error for the next module.
    IndependentError(oldErr, oldErr => {
      val clock = if (logger.isTraceEnabled) Stopwatch() else null

      val newErr = doDeriveInputError(
        input,
        reference,
        output,
        context,
        oldErr
      )

      if (clock != null) {
        logger.trace(
          f"$clock%s => deriveInputError(${oldErr.platform}%-4s) => $this%s"
        )
      }
      newErr
    })
  }

  protected def doDeriveWeightGradients(input:     Tensor,
                                        reference: Tensor,
                                        output:    Tensor,
                                        context:   PredictContext,
                                        error:     Tensor,
                                        sink:      ValueTensorBuffer)
  : Unit

  protected def doDeriveInputError(input:     Tensor,
                                   reference: Tensor,
                                   output:    Tensor,
                                   context:   PredictContext,
                                   error:     Tensor)
  : Tensor

}

trait TrainableLayerBuilder[TBuilder <: TrainableLayerBuilder[_]]
  extends LayerBuilder[TBuilder] {
}
