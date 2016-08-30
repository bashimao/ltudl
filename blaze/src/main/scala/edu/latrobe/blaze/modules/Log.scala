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
import edu.latrobe.blaze.TensorDependency._

/**
  * f(x_a) = log(x_a)
  *
  *       -1
  * f(x_a)   = exp(x_a)
  *
  * d f(x_a)    1
  * -------- = ---
  *  d x_a     x_a
  *
  *   d f(x_a)
  * ----------- = 0
  * d x_b, a!=b
  *
  *
  *            ---
  * D f(x_a)   \   d f(x_a)    1
  * -------- = /   -------- = ---
  *  D x_a     ---  d x_i     x_a
  *             i
  *
  */
abstract class Log
  extends NonTrainableMapLayer[LogBuilder]
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

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = {
    error :/= input
    error
  }

}

final class LogBuilder
  extends NonTrainableMapLayerBuilder[LogBuilder] {

  override def repr
  : LogBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LogBuilder]

  override protected def doCopy()
  : LogBuilder = LogBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = LogBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LogBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object LogBuilder extends ModuleVariantTable[LogBuilder] {

  register(2, Log_JVM_ApacheCommons_Description)
  register(4, Log_JVM_Baseline_Description)

  final def apply()
  : LogBuilder = new LogBuilder

}
