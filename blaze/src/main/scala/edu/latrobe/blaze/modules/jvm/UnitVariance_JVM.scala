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

abstract class UnitVariance_JVM
  extends UnitVariance
    with MapLayer_JVM[UnitVarianceBuilder] {

  final val (runningVariance, runningVarianceReference) = {
    val ref = builder.runningVarianceDevReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[RealArrayTensor]
      (result, None)
    }
    else {
      val result = RealArrayTensor.zeros(runningVarianceLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  override def refresh(): Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictForUnitTraining(input:        Tensor,
                                                        learningRate: Real)
  : (RealArrayTensor, PredictContext) = {
    val out = input.toRealArrayTensor
    val ctx = doPredictForUnitTraining(out, learningRate)
    (out, ctx)
  }

  protected def doPredictForUnitTraining(output:       RealArrayTensor,
                                         learningRate: Real)
  : PredictContext

  final override protected def doPredictForUnitInference(inPlaceAllowed: Boolean,
                                                         input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForUnitInference(out)
    out
  }

  protected def doPredictForUnitInference(output: RealArrayTensor)
  : Unit

  final override protected def doPredictForChannelTraining(input:        Tensor,
                                                           learningRate: Real)
  : (RealArrayTensor, PredictContext) = {
    val out = input.toRealArrayTensor
    val ctx = doPredictForChannelTraining(out, learningRate)
    (out, ctx)
  }

  protected def doPredictForChannelTraining(output:       RealArrayTensor,
                                            learningRate: Real)
  : PredictContext

  final override protected def doPredictForChannelInference(inPlaceAllowed: Boolean,
                                                            input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForChannelInference(out)
    out
  }

  protected def doPredictForChannelInference(output: RealArrayTensor)
  : Unit

  final override protected def doPredictForSampleTraining(input:        Tensor,
                                                          learningRate: Real)
  : (RealArrayTensor, PredictContext) = {
    val out = input.toRealArrayTensor
    val ctx = doPredictForSampleTraining(out, learningRate)
    (out, ctx)
  }

  protected def doPredictForSampleTraining(output:       RealArrayTensor,
                                           learningRate: Real)
  : PredictContext

  final override protected def doPredictForSampleInference(inPlaceAllowed: Boolean,
                                                           input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForSampleInference(out)
    out
  }

  protected def doPredictForSampleInference(output: RealArrayTensor)
  : Unit

  final override protected def doPredictForBatchTraining(input:        Tensor,
                                                         learningRate: Real)
  : (RealArrayTensor, PredictContext) = {
    val out = input.toRealArrayTensor
    val ctx = doPredictForBatchTraining(out, learningRate)
    (out, ctx)
  }

  protected def doPredictForBatchTraining(output:       RealArrayTensor,
                                          learningRate: Real)
  : PredictContext

  final override protected def doPredictForBatchInference(inPlaceAllowed: Boolean,
                                                          input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForBatchInference(out)
    out
  }

  protected def doPredictForBatchInference(output: RealArrayTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputErrorForUnit(input:   Tensor,
                                                         context: PredictContext,
                                                         error:   Tensor)
  : RealArrayTensor = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForUnit(inp, context, err)

    // Cleanup...
    if (inp ne input) {
      inp.close()
    }
    err
  }

  protected def doDeriveInputErrorForUnit(input:   RealArrayTensor,
                                          context: PredictContext,
                                          error:   RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorForChannel(input:   Tensor,
                                                            context: PredictContext,
                                                            error:   Tensor)
  : RealArrayTensor = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForChannel(inp, context, err)

    // Cleanup...
    if (inp ne input) {
      inp.close()
    }
    err
  }

  protected def doDeriveInputErrorForChannel(input:   RealArrayTensor,
                                             context: PredictContext,
                                             error:   RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorForSample(input:   Tensor,
                                                           context: PredictContext,
                                                           error:   Tensor)
  : RealArrayTensor = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForSample(inp, context, err)

    // Cleanup...
    if (inp ne input) {
      inp.close()
    }
    err
  }

  protected def doDeriveInputErrorForSample(input:   RealArrayTensor,
                                            context: PredictContext,
                                            error:   RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorForBatch(input:   Tensor,
                                                          context: PredictContext,
                                                          error:   Tensor)
  : RealArrayTensor = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForBatch(inp, context, err)

    // Cleanup...
    if (inp ne input) {
      inp.close()
    }
    err
  }

  protected def doDeriveInputErrorForBatch(input:   RealArrayTensor,
                                           context: PredictContext,
                                           error:   RealArrayTensor)
  : Unit

}
