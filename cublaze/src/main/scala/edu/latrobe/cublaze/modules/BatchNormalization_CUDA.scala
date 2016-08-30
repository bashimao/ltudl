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

package edu.latrobe.cublaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.cublaze._

abstract class BatchNormalization_CUDA
  extends BatchNormalization
    with MapLayer_CUDA[BatchNormalizationBuilder] {

  final override val (runningMean, runningMeanReference) = {
    val ref = builder.runningMeanReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[CUDARealTensor]
      (result, None)
    }
    else {
      val result = CUDARealTensor.zeros(device, runningMeanLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  final override val (runningVariance, runningVarianceReference) = {
    val ref = builder.runningVarianceReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[CUDARealTensor]
      (result, None)
    }
    else {
      val result = CUDARealTensor.zeros(device, runningVarianceLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  final override val (filter, filterReference) = {
    val ref = builder.filterReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[CUDARealTensor]
      (result, None)
    }
    else {
      val result = CUDARealTensor.zeros(device, filterLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  final override val (bias, biasReference) = {
    val ref = builder.biasReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[CUDARealTensor]
      (result, None)
    }
    else {
      val result = CUDARealTensor.zeros(device, biasLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  override protected def doClose()
  : Unit = {
    if (biasReference.isDefined) {
      bias.close()
    }
    if (filterReference.isDefined) {
      filter.close()
    }
    if (runningVarianceReference.isDefined) {
      runningVariance.close()
    }
    if (runningMeanReference.isDefined) {
      runningMean.close()
    }
    super.doClose()
  }

  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override def refresh()
  : Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictForTraining(input:        Tensor,
                                                    learningRate: Real)
  : (CUDARealTensor, PredictContext) = {
    val out = input.toCUDARealTensor(device)
    val ctx = doPredictForTraining(out, learningRate)
    (out, ctx)
  }

  protected def doPredictForTraining(output:       CUDARealTensor,
                                     learningRate: Real)
  : PredictContext

  final override protected def doPredictForInference(input: Tensor)
  : CUDARealTensor = {
    val out = input.toCUDARealTensor(device)
    doPredictForInference(out)
    out
  }

  protected def doPredictForInference(output: CUDARealTensor)
  : Unit

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : CUDARealTensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveWeightGradients(input:      Tensor,
                                                       context:    PredictContext,
                                                       error:      Tensor,
                                                       filterSink: Option[ValueTensor],
                                                       biasSink:   Option[ValueTensor])
  : Unit = {
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)

    val cudaFilterSink = {
      filterSink.map(
        _.asOrToCUDARealTensor(device)
      ).getOrElse(
        CUDARealTensor.zeros(device, filterLayout)
      )
    }
    val cudaBiasSink = {
      biasSink.map(
        _.asOrToCUDARealTensor(device)
      ).getOrElse(
        CUDARealTensor.zeros(device, biasLayout)
      )
    }

    // Compute gradients.
    doDeriveWeightGradients(
      inp,
      context,
      err,
      cudaFilterSink,
      cudaBiasSink
    )

    filterSink.foreach(filterSink => {
      if (filterSink ne cudaFilterSink) {
        cudaFilterSink.copyTo(filterSink)
        cudaFilterSink.close()
      }
    })
    biasSink.foreach(biasSink => {
      if (biasSink ne cudaBiasSink) {
        cudaBiasSink.copyTo(biasSink)
        cudaBiasSink.close()
      }
    })
    if (err ne error) {
      // See doDeriveInputError for explanation.
      error := err
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveWeightGradients(input:      CUDARealTensor,
                                        context:    PredictContext,
                                        error:      CUDARealTensor,
                                        filterSink: CUDARealTensor,
                                        biasSink:   CUDARealTensor)
  : Unit

  final override protected def doDeriveInputError(input:   Tensor,
                                                  context: PredictContext,
                                                  error:   Tensor)
  : Tensor = {
    // So far the only CUDA implementation we have gradients computation
    // performs all gradient computation in one go. In place?!
    error
  }

}
