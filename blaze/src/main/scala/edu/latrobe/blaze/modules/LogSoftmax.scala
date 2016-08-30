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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.modules.generic._

/**
  * Log space probability of the activation being a class label:
  *
  * Remark: In log space multiplication becomes addition:
  *
  *               ---
  *     ---       \
  * log | | x_i = /    log(x_i)
  *     | |       ---
  *      i         i
  *
  * logSoftmax(x_a) = log(softmax(x_a))
  *
  *
  * d logSoftmax(x_a)        1         d softmax(x_a)
  * ----------------- = ------------ * --------------
  *       d x_a         softmax(x_a)       d x_a
  *
  *                          1
  *                   = ------------ * softmax(x_a) (1 - softmax(x_a))
  *                     softmax(x_a)
  *
  *                   = 1 - softmax(x_a)
  *
  * d logSoftmax(x_a)        1         d softmax(x_a)
  * ----------------- = ------------ * --------------
  *   d x_b, b != a     softmax(x_a)       d x_b
  *
  *                          1
  *                   = ------------ -softmax(x_a) softmax(x_b)
  *                     softmax(x_a)
  *
  *                   = -1 * softmax(x_b)
  *
  *                   = -softmax(x_b)
  *
  *                     ---
  * D logSoftmax(x_a)   \   d logSoftmax(x_a)
  * ----------------- = /   ----------------- di
  *      D x_a          ---       d x_i
  *                      i
  *
  *                                             ----
  *                                             \
  *                   = (1 - softmax(x_a)) da + /    -softmax(x_i) di
  *                                             ----
  *                                             i!=a
  *
  *                          ---
  *                          \
  *                   = da - /   softmax(x_i) di
  *                          ---
  *                           i
  *
  * However, this is right. But we also have to consider that d is in log space.
  * (Note: Look at the proof for the ClassLLConstraint.)
  *
  *                                       ---
  *                                       \
  *                   = da - softmax(x_a) /    di
  *                                       ---
  *                                        i
  *
  *                                               ---
  *                                               \
  *                   = da - exp(logSoftmax(x_a)) /   di
  *                                               ---
  *                                                i
  **/
abstract class LogSoftmax
  extends NonTrainableMapLayer[LogSoftmaxBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredict(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(output, error)

  protected def doDeriveInputError(output: Tensor,
                                   error:  Tensor)
  : Tensor

}

final class LogSoftmaxBuilder
  extends NonTrainableMapLayerBuilder[LogSoftmaxBuilder] {

  override def repr
  : LogSoftmaxBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LogSoftmaxBuilder]


  override protected def doCopy()
  : LogSoftmaxBuilder = LogSoftmaxBuilder()


  // ---------------------------------------------------------------------------
  //   Statistics
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = LogSoftmaxBuilder.outputPlatformFor(this, hints)


  // ---------------------------------------------------------------------------
  //    Weights buffer handling related.
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LogSoftmaxBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object LogSoftmaxBuilder
  extends ModuleVariantTable[LogSoftmaxBuilder] {

  register( 2, LogSoftmax_JVM_Baseline_Description)
  register(64, LogSoftmax_Generic_Baseline_Description)

  final def apply(): LogSoftmaxBuilder = new LogSoftmaxBuilder

}
