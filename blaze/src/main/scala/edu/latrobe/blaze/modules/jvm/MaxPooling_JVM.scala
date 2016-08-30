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

package edu.latrobe.blaze.modules.jvm

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.TensorDependency._

abstract class MaxPooling_JVM
  extends MaxPooling {

  final override lazy val outputPlatform
  : JVM.type = JVM


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode: Mode, input: Tensor)
  : (Tensor, PredictContext) = {
    val inp       = input.asOrToRealArrayTensor
    val inpLayout = inp.layout
    val inpSize   = inpLayout.size
    val outSize   = kernel.outputSizeFor(inpSize, inpSize.noChannels)
    val outLayout = inpLayout.derive(outSize)
    val out       = RealArrayTensor.zeros(outLayout)
    val ctx       = doPredict(mode, inp, out)

    // Cleanup.
    if (inp ne input) {
      inp.close()
    }
    (out, ctx)
  }

  protected def doPredict(mode:   Mode,
                          input:  RealArrayTensor,
                          output: RealArrayTensor)
  : PredictContext


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputError(input:   Tensor,
                                                  output:  Tensor,
                                                  context: PredictContext,
                                                  error:   Tensor)
  : Tensor = context match {
    case context: MaxPooling_JVM_Context =>
      val oldErr = error.asOrToRealArrayTensor
      val newErr = RealArrayTensor.zeros(context.inputLayout)

      doDeriveInputError(context, oldErr, newErr)

      if (oldErr ne error) {
        oldErr.close()
      }
      newErr

    case _ =>
      throw new MatchError(context)
  }

  protected def doDeriveInputError(context:  PredictContext,
                                   oldError: RealArrayTensor,
                                   newError: RealArrayTensor)
  : Unit

}

abstract class MaxPooling_JVM_Context
  extends PredictContext {

  def inputLayout: IndependentTensorLayout

}
