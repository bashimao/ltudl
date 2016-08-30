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

final class Normalization_JVM_Baseline(override val builder:        NormalizationBuilder,
                                       override val inputHints:     BuildHints,
                                       override val seed:           InstanceSeed,
                                       override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Normalization_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  @inline
  private def doPredictForTraining(bufferIndex:    Int,
                                   muBuffer:       Array[Real],
                                   sigmaInvBuffer: Array[Real],
                                   learningRate:   Real,
                                   output:         Array[Real], offset: Int, stride: Int,
                                   length:         Int)
  : Unit = {
    // Compute current mean and variance.
    val mv = MeanAndVariance()
    ArrayEx.foreach(
      output, offset, stride,
      length
    )(mv.update)

    // Subtract mu.
    val mu = mv.mean
    ArrayEx.add(
      output, offset, stride,
      -mu,
      length
    )
    muBuffer(bufferIndex) = mu

    // Update running mu.
    if (runningMeanReference.isDefined) {
      val rm = runningMean.values
      rm(bufferIndex) = MathMacros.lerp(rm(bufferIndex), mu, learningRate)
    }

    // Divide by standard deviation.
    val popSigmaInv = Real.one / mv.populationStdDev(epsilon)
    ArrayEx.multiply(
      output, offset, stride,
      popSigmaInv,
      length
    )
    sigmaInvBuffer(bufferIndex) = popSigmaInv

    // Update running standard deviation.
    if (runningVarianceReference.isDefined) {
      val sigmaSq = mv.sampleVariance
      val rv      = runningVariance.values
      rv(bufferIndex) = MathMacros.lerp(rv(bufferIndex), sigmaSq, learningRate)
    }
  }

  @inline
  private def doPredictForInference(bufferIndex: Int,
                                    output:      Array[Real], offset: Int, stride: Int,
                                    length:      Int)
  : Unit = {
    val rm = runningMean.values
    val rv = runningVariance.values

    val mu = rm(bufferIndex)
    val sI = Real(1.0 / Math.sqrt(rv(bufferIndex) + epsilon))
    ArrayEx.transform(
      output, offset, stride,
      length
    )(x => (x - mu) * sI)
  }

  override protected def doPredictForUnitTraining(output:       RealArrayTensor,
                                                  learningRate: Real)
  : Normalization_JVM_Baseline_Context = {
    val mu       = new Array[Real](runningMean.values.length)
    val sigmaInv = new Array[Real](runningVariance.values.length)
    output.foreachUnit((offset, stride, length) => {
      doPredictForTraining(
        offset,
        mu,
        sigmaInv,
        learningRate,
        output.values, offset, stride,
        length
      )
    })
    Normalization_JVM_Baseline_Context(mu, sigmaInv)
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
  : Normalization_JVM_Baseline_Context = {
    val mu       = new Array[Real](runningMean.values.length)
    val sigmaInv = new Array[Real](runningVariance.values.length)
    output.foreachChannel((offset, stride, length) => {
      doPredictForTraining(
        offset,
        mu,
        sigmaInv,
        learningRate,
        output.values, offset, stride,
        length
      )
    })
    Normalization_JVM_Baseline_Context(mu, sigmaInv)
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
  : Normalization_JVM_Baseline_Context = {
    val mu       = new Array[Real](runningMean.values.length)
    val sigmaInv = new Array[Real](runningVariance.values.length)
    output.foreachSamplePair((i, offset, length) => {
      doPredictForTraining(
        i,
        mu,
        sigmaInv,
        learningRate,
        output.values, offset, 1,
        length
      )
    })
    Normalization_JVM_Baseline_Context(mu, sigmaInv)
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
  : Normalization_JVM_Baseline_Context = {
    val mu       = new Array[Real](runningMean.values.length)
    val sigmaInv = new Array[Real](runningVariance.values.length)
    doPredictForTraining(
      0,
      mu,
      sigmaInv,
      learningRate,
      output.values, 0, 1,
      output.values.length
    )
    Normalization_JVM_Baseline_Context(mu, sigmaInv)
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
                                 mu:       Real,
                                 sigmaInv: Real,
                                 error:    Array[Real], offset: Int, stride: Int,
                                 length:   Int)
  : Unit = {
    val a = ArrayEx.sum(
      error, offset, stride,
      length
    ) / length

    val b = ArrayEx.foldLeft(
      Real.zero,
      error, offset, stride,
      input, offset, stride,
      length
    )((res, err, inp) => res + err * (inp - mu)) / length * sigmaInv * sigmaInv

    ArrayEx.transform(
      error, offset, stride,
      input, offset, stride,
      length
    )((err, inp) => (err - a + (mu - inp) * b) * sigmaInv)
  }

  override protected def doDeriveInputErrorForUnit(input:   RealArrayTensor,
                                                   context: PredictContext,
                                                   error:   RealArrayTensor)
  : Unit = context match {
    case Normalization_JVM_Baseline_Context(mu, sigmaInv) =>
      error.foreachUnit((offset, stride, length) => {
        doDeriveInputError(
          input.values,
          mu(offset),
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
    case Normalization_JVM_Baseline_Context(mu, sigmaInv) =>
      error.foreachChannel((offset, stride, length) => {
        doDeriveInputError(
          input.values,
          mu(offset),
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
    case Normalization_JVM_Baseline_Context(mu, sigmaInv) =>
     error.foreachSamplePair((i, offset, length) => {
        doDeriveInputError(
          input.values,
          mu(offset),
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
    case Normalization_JVM_Baseline_Context(mu, sigmaInv) =>
      doDeriveInputError(
        input.values,
        mu(0),
        sigmaInv(0),
        error.values, 0, 1,
        error.values.length
      )

    case _ =>
      throw new MatchError(context)
  }

}

final case class Normalization_JVM_Baseline_Context(mu:       Array[Real],
                                                    sigmaInv: Array[Real])
  extends PredictContext {
}

object Normalization_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[NormalizationBuilder] {

  override def build(builder:        NormalizationBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Normalization_JVM_Baseline = new Normalization_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}

/*
// 20. Multichannel, No Kernel
final class NormalizationEx(desc:                     NormalizationExDesc,
                            override val bindContext: BindContext)
  extends NormalizationLike(desc) {

  import desc.noChannels


  private def predictInputCallback(input: DVec, muSig: (DVec, DVec)): Unit = {
    val tmp = input.asDenseMatrix.reshape(
      noChannels, input.length / noChannels, View.Require
    )(::, *)
    tmp :*= muSig._2
    tmp += muSig._1
  }

  /*out.tags.get(id) match {
      case Some(NormalizationExTagForSample(mu, sigma)) =>
        val outB = out.values.asDenseMatrix.reshape(
          noChannels, out.values.length / noChannels, View.Require
        )(::, *)
        outB :*= sigma
        outB += mu
      case _ =>
        throw new UnsupportedOperationException
    }
    */
  override def predictInput(mode: Mode, output: DSampleAct)
  : DSampleAct = output.tags.get(id) match {
    case Some(NormalizationExTagForSample(muSig)) =>
      val input = output.copyValues
      predictInputCallback(input.values, muSig)
      input
    case _ =>
      throw new UnsupportedOperationException
  }

  /*
    out.tags.get(id) match {
    case Some(NormalizationExTagForBatch(mu, sigma)) =>
      var i0 = 0
      while (i0 < out.values.rows) {
        val i1 = i0 + noChannels
        val tmpB = out.values(i0 until i1, ::)
        tmpB :*= sigma
        tmpB += mu
        i0 = i1
      }
    case _ =>
      throw new UnsupportedOperationException
    }
    */
  override def predictInput(mode: Mode, output: DBatchAct)
  : DBatchAct = output.tags.get(id) match {
    case Some(NormalizationExTagForBatch(muSig)) =>
      val input = output.copyValues
      cfor(0)(_ < muSig.length, _ + 1)(
        i => predictInputCallback(input.values(::, i), muSig(i))
      )
      input
    case _ =>
      throw new UnsupportedOperationException
  }

  private def deriveInputErrorCallback(error: DVec,
                                       input: DVec,
                                       mu:    Real,
                                       sig:   Real)
  : Unit = error.transformEx(input)((err, inp) => {
    // Nominator.
    val a = sig - sig / error.length
    val b = {
      val tmp = inp - mu
      (tmp * tmp) / (sig * (error.length - 1))
    }
    // Denominator.
    err * (a - b) / (sig * sig)
  })

  private def deriveInputError(error: DVec, input: DVec, mu: DVec, sig: DVec)
  : Unit = {
    val n         = error.length / noChannels
    val errSlices = error.asDenseMatrix.reshape(noChannels, n, View.Require)
    val inpSlices = error.asDenseMatrix.reshape(noChannels, n, View.Require)

    cfor(0)(_ < noChannels, _ + 1)(i => deriveInputErrorCallback(
      errSlices(i, ::).t,
      inpSlices(i, ::).t,
      mu.unsafeValueAt(i),
      sig.unsafeValueAt(i)
    ))

    /*
    // Nominator.
    // TODO: Avoid this allocation!
    val tmp: DMat = rawValuesB - mu
    tmp :*= tmp
    val sigmaTmp: DVec = sigma * (Real.one - n)
    val tmpB = tmp(::, *)
    tmpB :/= sigmaTmp

    sigmaTmp := sigma
    sigmaTmp *= Real.one - Real.one / n
    tmpB += sigmaTmp
    // Denominator.
    sigmaTmp := sigma
    sigmaTmp :*= sigmaTmp
    tmpB :/= sigmaTmp

    tmp.flatten(View.Require)
    */

  }

  override def deriveInputError(mode:   Mode,
                                error:  DVec,
                                output: DSampleAct,
                                input:  DSampleAct)
  : DVec = output.tags.get(id) match {
    case Some(NormalizationExTagForSample((mu, sig))) =>
      //err :*= gradient(raw.values, mu, sigma)
      deriveInputError(error, input.values, mu, sig)
      error
    case _ =>
      throw new UnsupportedOperationException
  }

  override def deriveInputError(mode:   Mode,
                                error:  DMat,
                                output: DBatchAct,
                                input:  DBatchAct)
  : DMat = output.tags.get(id) match {
    case Some(NormalizationExTagForBatch(muSig)) =>
      /*
      var i = 0
      while (i < raw.values.cols) {
        err(::, i) :*= gradient(raw.values(::, i), mu(::, i), sigma(::, i))
        i += 1
      }
      */
      cfor(0)(_ < error.cols, _ + 1)(i => {
        val (mu, sig) = muSig(i)
        deriveInputError(error(::, i), input.values(::, i), mu, sig)
      })
      error
    case _ =>
      throw new UnsupportedOperationException
  }

}

final class NormalizationExDesc(val          noChannels:     Int,
                                override val regularization: Real)
  extends NormalizationDescLike {
  debug_req(noChannels >= 1)
  debug_req(regularization >= Real.zero)

  override def noInputs: Int = -noChannels

  override def toStringEx
  : String = f"NormalizeEx[$noChannels%d, $regularization%.4e]"

  override def supports(size: Int): Boolean = size % noChannels == 0

  override def bind(context: BindContext)
  : Layer = new NormalizationEx(this, context)

}
*/
