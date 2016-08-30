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

final class UnitVariance_JVM_Baseline(override val builder:        UnitVarianceBuilder,
                                      override val inputHints:     BuildHints,
                                      override val seed:           InstanceSeed,
                                      override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends UnitVariance_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  @inline
  private def doPredictForTraining(bufferIndex:    Int,
                                   sigmaInvBuffer: Array[Real],
                                   learningRate:   Real,
                                   output:         Array[Real], offset: Int, stride: Int,
                                   length:         Int)
  : Unit = {
    // Compute variance.
    val varAcc = ArrayEx.l2NormSq(
      output, offset, stride,
      length
    )

    // Scale by population standard deviation.
    val popSigmaInv = Real(1.0 / Math.sqrt(varAcc / output.length + epsilon))
    ArrayEx.multiply(
      output, offset, stride,
      popSigmaInv,
      length
    )
    sigmaInvBuffer(bufferIndex) = popSigmaInv

    // Update running variance.
    if (runningVarianceReference.isDefined) {
      val sigmaSq = varAcc / (output.length - 1)
      val rv      = runningVariance.values
      rv(bufferIndex) = MathMacros.lerp(rv(bufferIndex), sigmaSq, learningRate)
    }
  }

  @inline
  private def doPredictForInference(bufferIndex: Int,
                                    output:      Array[Real], offset: Int, stride: Int,
                                    length:      Int)
  : Unit = {
    // Multiply sigma.
    val rv = runningVariance.values
    ArrayEx.multiply(
      output, offset, stride,
      Real(1.0 / Math.sqrt(rv(bufferIndex) + epsilon)),
      length
    )
  }

  override protected def doPredictForUnitTraining(output:       RealArrayTensor,
                                                  learningRate: Real)
  : UnitVariance_JVM_Baseline_Context = {
    val sigmaInv = new Array[Real](runningVariance.values.length)
    output.foreachUnit((offset, stride, length) => {
      doPredictForTraining(
        offset,
        sigmaInv,
        learningRate,
        output.values, offset, stride,
        length
      )
    })
    UnitVariance_JVM_Baseline_Context(sigmaInv)
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
  : UnitVariance_JVM_Baseline_Context = {
    val sigmaInv = new Array[Real](runningVariance.values.length)
    output.foreachChannel((offset, stride, length) => {
      doPredictForTraining(
        offset,
        sigmaInv,
        learningRate,
        output.values, offset, stride,
        length
      )
    })
    UnitVariance_JVM_Baseline_Context(sigmaInv)
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
  : UnitVariance_JVM_Baseline_Context = {
    val sigmaInv = new Array[Real](runningVariance.values.length)
    output.foreachSamplePair((i, offset, length) => {
      doPredictForTraining(
        i,
        sigmaInv,
        learningRate,
        output.values, offset, 1,
        length
      )
    })
    UnitVariance_JVM_Baseline_Context(sigmaInv)
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
  : UnitVariance_JVM_Baseline_Context = {
    val sigmaInv = new Array[Real](runningVariance.values.length)
    doPredictForTraining(
      0,
      sigmaInv,
      learningRate,
      output.values, 0, 1,
      output.values.length
    )
    UnitVariance_JVM_Baseline_Context(sigmaInv)
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
  private def doDeriveInputError(input:    Array[Real],
                                 sigmaInv: Real,
                                 error:    Array[Real], offset: Int, stride: Int,
                                 length:   Int)
  : Unit = {
    val tmp = ArrayEx.dot(
      error, offset, stride,
      input, offset, stride,
      length
    ) / length * sigmaInv * sigmaInv
    ArrayEx.transform(
      error, offset, stride,
      input, offset, stride,
      length
    )((err, inp) => (err - inp * tmp) * sigmaInv)
  }


  override protected def doDeriveInputErrorForUnit(input:   RealArrayTensor,
                                                   context: PredictContext,
                                                   error:   RealArrayTensor)
  : Unit = context match {
    case UnitVariance_JVM_Baseline_Context(sigmaInv) =>
      error.foreachUnit((offset, stride, length) => {
        doDeriveInputError(
          input.values,
          sigmaInv(offset),
          error.values, offset, stride,
          length
        )
      })
    case _ =>
      throw new MatchError(context)
  }

  override protected def doDeriveInputErrorForChannel(input:   RealArrayTensor,
                                                      context: PredictContext,
                                                      error:   RealArrayTensor)
  : Unit = context match {
    case UnitVariance_JVM_Baseline_Context(sigmaInv) =>
      error.foreachChannel((offset, stride, length) => {
        doDeriveInputError(
          input.values,
          sigmaInv(offset),
          error.values, offset, stride,
          length
        )
      })
    case _ =>
      throw new MatchError(context)
  }

  override protected def doDeriveInputErrorForSample(input:   RealArrayTensor,
                                                     context: PredictContext,
                                                     error:   RealArrayTensor)
  : Unit = context match {
    case UnitVariance_JVM_Baseline_Context(sigmaInv) =>
      error.foreachSamplePair((i, offset, length) => {
        doDeriveInputError(
          input.values,
          sigmaInv(i),
          error.values, offset, 1,
          length
        )
      })
    case _ =>
      throw new MatchError(context)
  }

  override protected def doDeriveInputErrorForBatch(input:   RealArrayTensor,
                                                    context: PredictContext,
                                                    error:   RealArrayTensor)
  : Unit = context match {
    case UnitVariance_JVM_Baseline_Context(sigmaInv) =>
      doDeriveInputError(
        input.values,
        sigmaInv(0),
        error.values, 0, 1,
        error.values.length
      )
    case _ =>
      throw new MatchError(context)
  }

}

final case class UnitVariance_JVM_Baseline_Context(sigmaInv: Array[Real])
  extends PredictContext {
}

object UnitVariance_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[UnitVarianceBuilder] {

  override def build(builder:        UnitVarianceBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : UnitVariance_JVM_Baseline = new UnitVariance_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
