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

abstract class ZeroMean_JVM
  extends ZeroMean
    with MapLayer_JVM[ZeroMeanBuilder] {

  final val (runningMean, runningMeanReference) = {
    val ref = builder.runningMeanReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[RealArrayTensor]
      (result, None)
    }
    else {
      val result = RealArrayTensor.zeros(runningMeanLayout)
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
  final override protected def doPredictForUnitTraining(inPlaceAllowed: Boolean,
                                                        input:          Tensor,
                                                        learningRate:   Real)
  : (Tensor, PredictContext) = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForUnitTraining(out, learningRate)
    (out, EmptyContext)
  }

  protected def doPredictForUnitTraining(output:       RealArrayTensor,
                                         learningRate: Real)
  : Unit

  final override protected def doPredictForUnitInference(inPlaceAllowed: Boolean,
                                                         input:          Tensor)
  : Tensor = {
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

  final override protected def doPredictForChannelTraining(inPlaceAllowed: Boolean,
                                                           input:          Tensor,
                                                           learningRate:   Real)
  : (Tensor, PredictContext) = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForChannelTraining(out, learningRate)
    (out, EmptyContext)
  }

  protected def doPredictForChannelTraining(output:       RealArrayTensor,
                                            learningRate: Real)
  : Unit

  final override protected def doPredictForChannelInference(inPlaceAllowed: Boolean,
                                                            input:          Tensor)
  : Tensor = {
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

  final override protected def doPredictForSampleTraining(inPlaceAllowed: Boolean,
                                                          input:          Tensor,
                                                          learningRate:   Real)
  : (Tensor, PredictContext) = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForSampleTraining(out, learningRate)
    (out, EmptyContext)
  }

  protected def doPredictForSampleTraining(output:       RealArrayTensor,
                                           learningRate: Real)
  : Unit

  final override protected def doPredictForSampleInference(inPlaceAllowed: Boolean,
                                                           input:          Tensor)
  : Tensor = {
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

  final override protected def doPredictForBatchTraining(inPlaceAllowed: Boolean,
                                                         input:          Tensor,
                                                         learningRate:   Real)
  : (Tensor, PredictContext) = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictForBatchTraining(out, learningRate)
    (out, EmptyContext)
  }

  protected def doPredictForBatchTraining(output:       RealArrayTensor,
                                          learningRate: Real)
  : Unit

  final override protected def doPredictForBatchInference(inPlaceAllowed: Boolean,
                                                          input:          Tensor)
  : Tensor = {
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
  final override protected def doDeriveInputErrorForUnit(context: PredictContext,
                                                         error:   Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForUnit(context, err)
    err
  }

  protected def doDeriveInputErrorForUnit(context: PredictContext,
                                          error:   RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorForChannel(context: PredictContext,
                                                            error:   Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForChannel(context, err)
    err
  }

  protected def doDeriveInputErrorForChannel(context: PredictContext,
                                             error:   RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorForSample(context: PredictContext,
                                                           error:   Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForSample(context, err)
    err
  }

  protected def doDeriveInputErrorForSample(context: PredictContext,
                                            error:   RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorForBatch(context: PredictContext,
                                                          error:   Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorForBatch(context, err)
    err
  }

  protected def doDeriveInputErrorForBatch(context: PredictContext,
                                           error:   RealArrayTensor)
  : Unit

}
