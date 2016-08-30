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

final class ZeroMean_JVM_Baseline(override val builder:        ZeroMeanBuilder,
                                  override val inputHints:     BuildHints,
                                  override val seed:           InstanceSeed,
                                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ZeroMean_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  @inline
  private def doPredictForTraining(bufferIndex:  Int,
                                   learningRate: Real,
                                   output:       Array[Real], offset: Int, stride: Int,
                                   length:       Int)
  : Unit = {
    // Compute mu.
    val mu = ArrayEx.mean(
      output, offset, stride,
      length
    )

    // Subtract mu.
    ArrayEx.add(
      output, offset, stride,
      -mu,
      length
    )

    // Update running mu.
    if (runningMeanReference.isDefined) {
      val rm = runningMean.values
      rm(bufferIndex) = MathMacros.lerp(rm(bufferIndex), mu, learningRate)
    }
  }

  @inline
  private def doPredictForInference(bufferIndex: Int,
                                    output:      Array[Real], offset: Int, stride: Int,
                                    length:      Int)
  : Unit = {
    // Subtract mu.
    val rm = runningMean.values
    ArrayEx.add(
      output, offset, stride,
      -rm(bufferIndex),
      length
    )
  }

  override protected def doPredictForUnitTraining(output:       RealArrayTensor,
                                                  learningRate: Real)
  : Unit = {
    output.foreachUnit((offset, stride, length) => {
      doPredictForTraining(
        offset,
        learningRate,
        output.values, offset, stride,
        length
      )
    })
  }

  override protected def doPredictForUnitInference(output: RealArrayTensor)
  : Unit = {
    output.foreachUnit((offset, stride, length) => {
      doPredictForInference(
        offset,
        output.values, offset, stride,
        length
      )
    })
  }

  override protected def doPredictForChannelTraining(output:       RealArrayTensor,
                                                     learningRate: Real)
  : Unit = {
    output.foreachChannel((offset, stride, length) => {
      doPredictForTraining(
        offset,
        learningRate,
        output.values, offset, stride,
        length
      )
    })
  }

  override protected def doPredictForChannelInference(output: RealArrayTensor)
  : Unit = {
    output.foreachChannel((offset, stride, length) => {
      doPredictForInference(
        offset,
        output.values, offset, stride,
        length
      )
    })
  }

  override protected def doPredictForSampleTraining(output:       RealArrayTensor,
                                                    learningRate: Real)
  : Unit = {
    output.foreachSamplePair((i, offset, length) => {
      doPredictForTraining(
        i,
        learningRate,
        output.values, offset, 1,
        length
      )
    })
  }

  override protected def doPredictForSampleInference(output: RealArrayTensor)
  : Unit = {
    output.foreachSamplePair((i, offset, length) => {
      doPredictForInference(
        i,
        output.values, offset, 1,
        length
      )
    })
  }

  override protected def doPredictForBatchTraining(output:       RealArrayTensor,
                                                   learningRate: Real)
  : Unit = {
    doPredictForTraining(
      0,
      learningRate,
      output.values, 0, 1,
      output.values.length
    )
  }

  override protected def doPredictForBatchInference(output: RealArrayTensor)
  : Unit = {
    doPredictForInference(
      0,
      output.values, 0, 1,
      output.values.length
    )
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  @inline
  private def doDeriveInputError(error:  Array[Real], offset: Int, stride: Int,
                                 length: Int)
  : Unit = {
    val mu = ArrayEx.mean(
      error, offset, stride,
      length
    )
    ArrayEx.transform(
      error, offset, stride,
      length
    )(err => err * (err - mu))
  }

  override protected def doDeriveInputErrorForUnit(context: PredictContext,
                                                   error:   RealArrayTensor)
  : Unit = {
    error.foreachUnit((offset, stride, length) => {
      doDeriveInputError(
        error.values, offset, stride,
        length
      )
    })
  }

  override protected def doDeriveInputErrorForChannel(context: PredictContext,
                                                      error:   RealArrayTensor)
  : Unit = {
    error.foreachChannel((offset, stride, length) => {
      doDeriveInputError(
        error.values, offset, stride,
        length
      )
    })
  }

  override protected def doDeriveInputErrorForSample(context: PredictContext,
                                                     error:   RealArrayTensor)
  : Unit = {
    error.foreachSample((offset, length) => {
      doDeriveInputError(
        error.values, offset, 1,
        length
      )
    })
  }

  override protected def doDeriveInputErrorForBatch(context: PredictContext,
                                                    error:   RealArrayTensor)
  : Unit = {
    doDeriveInputError(
      error.values, 0, 1,
      error.values.length
    )
  }

}

object ZeroMean_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[ZeroMeanBuilder] {

  override def build(builder:        ZeroMeanBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ZeroMean_JVM_Baseline = new ZeroMean_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
