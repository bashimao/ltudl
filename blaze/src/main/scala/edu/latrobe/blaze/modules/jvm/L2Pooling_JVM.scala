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

abstract class L2Pooling_JVM
  extends L2Pooling {

  final override lazy val outputPlatform
  : JVM.type = JVM


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:  Mode,
                                         input: Tensor)
  : (RealArrayTensor, PredictContext) = {
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


  /**
    * Since we average on predict, we have errors regarding the averages of many
    * activations. Hence, we simply reverse the averaging process here.
    */
  final override protected def doDeriveInputError(input:   Tensor,
                                                  output:  Tensor,
                                                  context: PredictContext,
                                                  error:   Tensor)
  : RealArrayTensor = context match {
    case context: MeanPooling_JVM_Context =>
      val inp    = input.asOrToRealArrayTensor
      val out    = output.asOrToRealArrayTensor
      val oldErr = error.asOrToRealArrayTensor
      val newErr = RealArrayTensor.zeros(context.inputLayout)

      doDeriveInputError(inp, out, context, oldErr, newErr)

      if (oldErr ne error) {
        oldErr.close()
      }
      if (out ne output) {
        out.close()
      }
      if (inp ne input) {
        inp.close()
      }
      newErr

    case _ =>
      throw new MatchError(context)
  }

  protected def doDeriveInputError(input:    RealArrayTensor,
                                   output:   RealArrayTensor,
                                   context:  PredictContext,
                                   oldError: RealArrayTensor,
                                   newError: RealArrayTensor)
  : Unit

}
