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

/**
  * Softmax activations (probabilities)
  *
  *                  exp(x_a)
  * softmax(x_a) = ------------
  *                ---
  *                \
  *                /   exp(x_i)
  *                ---
  *                 i
  *
  *                                                           ( ---          )
  *                                                           ( \            )
  *                                                         d ( /   exp(x_i) )
  *                             ( ---          )              ( ---          )
  *                  d exp(x_a) ( \            )              (  i           )
  *                  ---------  ( /   exp(x_i) ) - exp(x_a) ------------------
  *                    d x_a    ( ---          )                   d x_a
  * d softmax(x_a)              (  i           )
  * -------------- = ---------------------------------------------------------
  *      d x_a                                         2
  *                                    ( ---          )
  *                                    ( \            )
  *                                    ( /   exp(x_i) )
  *                                    ( ---          )
  *                                    (  i           )
  *
  *                           ( ---          )
  *                           ( \            )
  *                  exp(x_a) ( /   exp(x_i) ) - exp(x_a) exp(x_a)
  *                           ( ---          )
  *                           (  i           )
  *                = -----------------------------------------------
  *                                                2
  *                                ( ---          )
  *                                ( \            )
  *                                ( /   exp(x_i) )
  *                                ( ---          )
  *                                (  i           )
  *
  *
  *                           ( ---          )
  *                           ( \            )
  *                  exp(x_a) ( /   exp(x_i) ) - exp(x_a) exp(x_a)
  *                           ( ---          )
  *                           (  i           )
  *                = ---------------------------------------------
  *                         ( ---          ) ( ---          )
  *                         ( \            ) ( \            )
  *                         ( /   exp(x_i) ) ( /   exp(x_i) )
  *                         ( ---          ) ( ---          )
  *                         (  i           ) (  i           )
  *
  *                = softmax(x_a) - softmax(x_a) softmax(x_a)
  *
  *                = softmax(x_a) (1 - softmax(x_a))
  *
  *                                                           ( ---          )
  *                                                           ( \            )
  *                                                         d ( /   exp(x_i) )
  *                             ( ---          )              ( ---          )
  *                  d exp(x_a) ( \            )              (  i           )
  *                  ---------  ( /   exp(x_i) ) - exp(x_a) ------------------
  *                    d x_b    ( ---          )                   d x_b
  * d softmax(x_a)              (  i           )
  * -------------- = ---------------------------------------------------------
  * d x_b, a != b                                      2
  *                                    ( ---          )
  *                                    ( \            )
  *                                    ( /   exp(x_i) )
  *                                    ( ---          )
  *                                    (  i           )
  *
  *
  *                  0 * exp(x_b) - exp(x_a) exp(x_b)
  *                = -------------------------------
  *                  ( ---          ) ( ---          )
  *                  ( \            ) ( \            )
  *                  ( /   exp(x_i) ) ( /   exp(x_i) )
  *                  ( ---          ) ( ---          )
  *                  (  i           ) (  i           )
  *
  *                = -softmax(x_a) softmax(x_b)
  *
  *                  ---
  * D softmax(x_a)   \   d softmax(x_a)
  * -------------- = /   -------------- di
  *     D x_a        ---      d x_i
  *                   i
  *
  *                                                                  ----
  *                                                                  \
  *                = (softmax(x_a) - softmax(x_a) softmax(x_a)) da + /    -softmax(x_a) softmax(x_i) di
  *                                                                  ----
  *                                                                  i!=a
  *
  *                                    ---
  *                                    \
  *                = softmax(x_a) da + /   -softmax(x_a) softmax(x_i) di
  *                                    ---
  *                                     i
  *
  *                                                 ---
  *                                                 \
  *                = softmax(x_a) da - softmax(x_a) /   softmax(x_i) di
  *                                                 ---
  *                                                  i
  *
  *                               (      ---                 )
  *                               (      \                   )
  *                = softmax(x_a) ( da - /   softmax(x_i) di )
  *                               (      ---                 )
  *                               (       i                  )
  *
  */
// TODO: Do a "fast" version.
abstract class Softmax
  extends NonTrainableMapLayer[SoftmaxBuilder]
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

  protected def doPredict(inPlaceAllowed: Boolean,
                          input:          Tensor)
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

final class SoftmaxBuilder
  extends NonTrainableMapLayerBuilder[SoftmaxBuilder] {

  override def repr
  : SoftmaxBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SoftmaxBuilder]

  override protected def doCopy()
  : SoftmaxBuilder = SoftmaxBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = SoftmaxBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SoftmaxBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object SoftmaxBuilder extends ModuleVariantTable[SoftmaxBuilder] {

  register(2, Softmax_JVM_ApacheCommons_Description)
  register(4, Softmax_JVM_Baseline_Description)

  final def apply()
  : SoftmaxBuilder = new SoftmaxBuilder

}
