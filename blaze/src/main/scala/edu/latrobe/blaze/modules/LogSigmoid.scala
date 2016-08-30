/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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
import edu.latrobe.blaze.modules.jvm._

/**
 *                 (  -x     )
 * Predict: f = -ln( e   + 1 )
 *                 (         )
 *
 *           -1     (  -y     )
 * Inverse: f  = -ln( e   - 1 )
 *                  (         )
 *
 *                     -x
 *           d f      e
 * Gradient: --- =  -------
 *           d x     -x
 *                  e   + 1
 *
 */
abstract class LogSigmoid
  extends NonTrainableMapLayer[LogSigmoidBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredict(input)
    (out, EmptyContext)
  }

  protected def doPredict(input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = doPredictInv(output)

  protected def doPredictInv(output: Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final protected def doDeriveInputError(input:     Tensor,
                                         reference: Tensor,
                                         output:    Tensor,
                                         context:   PredictContext,
                                         error:     Tensor)
  : Tensor = doDeriveInputError(input, context, error)

  protected def doDeriveInputError(input:   Tensor,
                                   context: PredictContext,
                                   error:   Tensor)
  : Tensor

}

final class LogSigmoidBuilder
  extends NonTrainableMapLayerBuilder[LogSigmoidBuilder] {

  override def repr
  : LogSigmoidBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LogSigmoidBuilder]


  override protected def doCopy()
  : LogSigmoidBuilder = LogSigmoidBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = LogSigmoidBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LogSigmoidBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object LogSigmoidBuilder
  extends ModuleVariantTable[LogSigmoidBuilder] {

  register(2, LogSigmoid_JVM_ApacheCommons_Description)
  register(4, LogSigmoid_JVM_Baseline_Description)

  final def apply(): LogSigmoidBuilder = new LogSigmoidBuilder

}