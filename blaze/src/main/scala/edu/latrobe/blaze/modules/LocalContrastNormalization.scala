/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules

import breeze.numerics._
import breeze.linalg.{DenseMatrix, DenseVector, axpy}
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.kernels._
import scala.util.hashing._

// 30. Single Channel, Kernel (optimized implementation TBD)
// 40. Multichannel, Kernel

// TODO: Add code that allows lateral normalization.
final class LocalContrastNormalization(override val builder:        LocalContrastNormalizationBuilder,
                                       override val inputHints:     BuildHints,
                                       override val seed:           InstanceSeed,
                                       override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[LocalContrastNormalizationBuilder]
    with NonTrainableLayer[LocalContrastNormalizationBuilder]
    with NonPenalizing {

  override val outputHints
  : BuildHints = inputHints.derive(JVM)

  val kernel
  : Kernel = builder.kernel

  val window
  : Window = builder.window

  require(window.noWeights == kernel.noValues)

  val epsilon
  : Real = builder.epsilon

  /**
   * Although it would not be necessary, I decided to expand the vectors
   * here directly. Reason: Speed and readiblity of further code!
   */
  /*
  // TODO: Check if the assessment above holds true.
  protected lazy val (wCounts: DenseVector[Int], wSums: DVec) = {
    // TODO: Make generating these luts is more efficient!
    val wc = DenseMatrix.zeros[Int](noChannels, kernel.noInputs)
    val ws = {
      if (window == null) {
        kernel.foreachValidPair((i, offset) => (j, offset) => wc(::, i) += 1)
        wc.map(Real(_))
      }
      else {
        val ws = DMat.zeros(wc.rows, wc.cols)
        kernel.foreachValidPair((i, offset) => (j, offset) => {
          val w = window(j)
          // TODO: Check whether these boundary conditions are sufficient.
          if (w != Real.zero) {
            wc(::, i) += 1
            ws(::, i) += w
          }
        })
        ws
      }
    }
    (wc.flatten(View.Require), ws.flatten(View.Require))
  }
  */

  protected def scaleSigma(q: DenseVector[Real], nSums: DenseVector[Int])
  : Unit = VectorEx.transform(q, nSums)((q, n) => {
    if (n > 1) {
      val variance = q / (n - 1)
      Math.max(sqrt(variance), epsilon)
    }
    else {
      epsilon
    }
  })

  protected def scaleSigma(q:     DenseVector[Real],
                           nSums: DenseVector[Int],
                           wSums: DenseVector[Real])
  : Unit = VectorEx.transform(q, nSums, wSums)((q, n, wn) => {
    if (n > 1) {
      val variance = q / wn * n / (n - 1)
      Math.max(sqrt(variance), epsilon)
    }
    else {
      epsilon
    }
  })

  /*
  override protected def doPredict(mode: ComputeMode, input: SampleTensor)
  : (SampleTensor, Any) = {
    val inp = input.values
    // TODO: Check if expanded n & wn is really faster!
    val n   = DenseVector.zeros[Int](inp.length)
    val wn  = if (windowFn == null) null else DVec.zeros(inp.length)
    val mu  = DVec.zeros(inp.length)
    val sig = DVec.zeros(inp.length)
    val tmp = DVec.zeros(input.size.noChannels)

    // Incrementally compute mu and sigma.
    if (windowFn == null) {
      kernel.foreachValidPairEx(input.size, tmp.length, (i0, i1, offset0) => {
        /*val dstRange = {
          val dst0 = i * noChannels
          dst0 until dst0 + noChannels
        }*/
        val dstRange = i0 until i1
        val muSlice  = mu(dstRange)
        val sigSlice = sig(dstRange)

        var wSum = 0
        Tuple2(
          (j0, j1, offset0, offset1) => {
            // x - mu
            /*val src = {
              val src0 = offset * noChannels
              inp(src0 until src0 + noChannels)
            }*/
            val src = inp(offset0 until offset1)
            //println(s"$i0,$i1 === $j0,$j1 === $offset0,$offset1")
            tmp := src
            tmp -= muSlice

            wSum += 1
            val wFac = Real.one / wSum

            // Update mu.
            axpy(wFac, tmp, muSlice)

            // Update sigma.
            tmp :*= tmp
            axpy(Real.one - wFac, tmp, sigSlice)
          },
          () => n(dstRange) := wSum
        )
      })
      scaleSigma(sig, n)
    }
    else {
      kernel.foreachValidPairIndexEx(input.size, tmp.length, (i0 ,i1, offset0) => {
        /*val dstRange = {
          val dst0 = i * noChannels
          dst0 until dst0 + noChannels
        }*/
        val dstRange = i0 until i1
        val muSlice  = mu(dstRange)
        val sigSlice = sig(dstRange)

        var nSum  = 0
        var wSum1 = Real.zero
        Tuple2(
          (j, offset0, offset1) => {
            // x - mu
            /*val src = {
              val src0 = offset * noChannels
              inp(src0 until src0 + noChannels)
            }*/
            val src = inp(offset0 until offset1)
            tmp := src
            tmp -= muSlice

            val wSum0 = wSum1
            val wFac = {
              val w = windowFn(j)
              wSum1 += w
              w / wSum1
            }
            nSum += 1

            // Update mu.
            axpy(wFac, tmp, muSlice)

            // Update sigma.
            //tmp :*= tmp
            //axpy(wFac * wSum0, tmp, sigSlice)
            val wTmp = Real.one - wFac
            sigSlice.transformEx(tmp)((sig, tmp) => {
              sig + tmp * tmp * wTmp
            })
          },
          () => {
            // TOOD: Make this a simple array operation!
            n(dstRange)  := nSum
            wn(dstRange) := wSum1
          }
        )
      })
      scaleSigma(sig, n, wn)
    }

    // Perform normalization.
    val out = inp - mu
    out :/= sig
    DenseSampleTensor(out, input.size) -> LocalContrastNormalizationMetaDataS(mu, sig, n, wn)
  }*/

  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val inp     = input.valuesMatrix
    val inpSize = input.layout.size
    // TODO: Check if expanded n & wn is really faster!
    val nSums   = DenseVector.zeros[Int](inp.rows)
    val wSums   = if (window == null) null else DenseVector.zeros[Real](inp.rows)
    val mu      = DenseMatrix.zeros[Real](inp.rows, inp.cols)
    val sigma   = DenseMatrix.zeros[Real](inp.rows, inp.cols)
    val tmp     = DenseMatrix.zeros[Real](inpSize.noChannels, inp.cols)

    // Incrementally compute mu and sigma.
    if (window == null) {
      kernel.foreachValidPairEx(inpSize, tmp.rows, (i0, i1, offset0) => {
        val dstRange   = Range(i0, i1)
        val muSlice    = mu(dstRange, ::)
        val sigmaSlice = sigma(dstRange, ::)

        var wSum = 0
        Tuple2(
          (j0, j1, offset0, offset1) => {
            // x - mu
            val src = inp(offset0 until offset1, ::)
            tmp := src
            tmp -= muSlice

            wSum += 1
            val wFac = Real.one / wSum

            // Update mu.
            //axpy(wFac, tmp, muSlice)
            MatrixEx.add(
              muSlice, wFac, tmp
            )
            //muSlice.transformEx(tmp)((mu, tmp) => mu + tmp * wFac)

            // Update sigma.
            //tmp :*= tmp
            //axpy(Real.one - wFac, tmp, sigSlice)
            //Native.axpy(Real.one - wFac, tmp, sigSlice)
            val wTmp = Real.one - wFac
            MatrixEx.transform(sigmaSlice, tmp)(
              (sig, tmp) => sig + tmp * tmp * wTmp
            )
          },
          () => nSums(dstRange) := wSum
        )
      })
      MatrixEx.foreachColumnVector(sigma)(
        scaleSigma(_, nSums)
      )
    }
    else {
      kernel.foreachValidPairIndexEx(inpSize, tmp.rows, (i0, i1, offset0) => {
        /*val dstRange = {
          val dst0 = i * noChannels
          dst0 until dst0 + noChannels
        }*/
        val dstRange = Range(i0, i1)
        val muSlice  = mu(dstRange, ::)
        val sigSlice = sigma(dstRange, ::)

        var nSum  = 0
        var wSum1 = Real.zero
        Tuple2(
          (j, offset0, offset1) => {
            // x - mu
            val src = inp(offset0 until offset1, ::)
            tmp := src
            tmp -= muSlice

            val wSum0 = wSum1
            val wFac = {
              val w = window(j)
              wSum1 += w
              w / wSum1
            }
            nSum += 1

            // Update mu.
            axpy(wFac, tmp, muSlice)

            // Update sigma.
            //tmp :*= tmp
            //axpy(wFac * wSum0, tmp, sigSlice)
            val wTmp = wFac * wSum0
            MatrixEx.transform(sigSlice, tmp)(
              (sig, tmp) => sig + tmp * tmp * wTmp
            )
          },
          () => {
            nSums(dstRange) := nSum
            wSums(dstRange) := wSum1
          }
        )
      })
      MatrixEx.foreachColumnVector(sigma)(
        scaleSigma(_, nSums, wSums)
      )
    }

    // Perform normalization.
    val out = inp - mu
    out :/= sigma
    val context = LocalContrastNormalizationContext(mu, sigma, nSums, wSums)
    (RealArrayTensor.derive(inpSize, out), context)
  }

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = context match {
    case LocalContrastNormalizationContext(mu, sigma, nSums, wSums) =>
      val inp = output.valuesMatrix :* sigma
      inp += mu
      RealArrayTensor.derive(output.layout.size, inp)
    case _ =>
      throw new MatchError(context)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  /*
  private lazy val deriveInputErrorFactors: (DVec, DVec) = {
    val w = window(kernel.localPairIndexOfCenterPair)

    val a = wSums.fastMap(wSum => Real.one - w / wSum)
    val b = wCounts.fastMapEx(wSums)(
      (n, wSum) => n * w / ((n - Real.one) * wSum)
    )
    (factor1, factor2)
  }*/

  protected def deriveInputError(err: DenseVector[Real],
                                 inp: DenseVector[Real],
                                 mu:  DenseVector[Real],
                                 sig: DenseVector[Real],
                                 n:   DenseVector[Int])
  : Unit = VectorEx.transform(err, inp, mu, sig, n)(
    (err, inp, mu, sig, n) => {
      val tmp = inp - mu
      val b = (tmp * tmp) / (sig * (n - 1))
      val a = sig - sig / n
      err * (a - b) / (sig * sig)
    }
  )

  protected def deriveInputError(err: DenseVector[Real],
                                 inp: DenseVector[Real],
                                 mu:  DenseVector[Real],
                                 sig: DenseVector[Real],
                                 n:   DenseVector[Int],
                                 wn:  DenseVector[Real],
                                 w:   Real)
  : Unit = VectorEx.transform(err, inp, mu, sig, n, wn)(
    (err, inp, mu, sig, n, wn) => {
      val tmp = inp - mu
      val b = (tmp * tmp * n * w) / (sig * (n - 1) * wn)
      val a = sig - sig * w / wn
      err * (a - b) / (sig * sig)
    }
  )

  /*
  override def deriveInputError(mode:      ComputeMode,
                                input:     SampleTensor,
                                output:    SampleTensor,
                                context:   Any,
                                error:     SampleTensor,
                                reference: SampleTensor)
  : SampleTensor = {
    val inp = input.values
    val err = error.values

    context match {
      case LocalContrastNormalizationMetaDataS(mu, sigma, n, wn) =>
        if (windowFn == null) {
          deriveInputError(err, inp, mu, sigma, n)
        }
        else {
          val w = windowFn(kernel.localPairNoOfCenterPair)
          deriveInputError(err, inp, mu, sigma, n, wn, w)
        }
      /*
      val tmp = DVec.ones(factor.length)
      tmp -= counts
      tmp :*= sig
      factor :/= tmp
      // TODO: Could avoid this!
      tmp := -Real.one
      tmp :/= counts
      tmp += Real.one
      tmp :*= sig
      factor += tmp
      // Denominator.
      factor :/= sig
      factor :/= sig
      err :*= factor
      */
      /*
      // 1st term.
      val tmp0 = sig :* gradientFactor._1
      // 2nd term.
      val tmp1 = raw.values - mu
      tmp1 :*= tmp1
      tmp1 :/= sig
      tmp1 :*= gradientFactor._2
      // Combine
      tmp0 -= tmp1
      tmp0 :/= sig
      tmp0 :/= sig
      */
      case _ =>
        throw new MatchError(context)
    }

    DenseSampleTensor(err, error.size)
  }*/

  override protected def doDeriveInputError(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor = {
    val inp = input.valuesMatrix
    val err = error.valuesMatrix

    context match {
      case LocalContrastNormalizationContext(mu, sigma, n, wn) =>
        if (window == null) {
          MatrixEx.foreachColumnVector(err, inp, mu, sigma)(
            deriveInputError(_, _, _, _, n)
          )
        }
        else {
          val w = window(kernel.localPairNoOfCenterPair)
          MatrixEx.foreachColumnVector(err, inp, mu, sigma)(
            deriveInputError(_, _, _, _, n, wn, w)
          )
        }
      /*// TODO: Could code this more elegant.
      // Nominator.
      val tmp = raw.values - mu
      tmp :*= tmp
      // val sigmaTmp: DVec = sigma * (Real.one - x.cols)
      val sigTmp  = DMat.ones(tmp.rows, tmp.cols)
      val sigTmpB = sigTmp(::, *)
      sigTmpB  -= counts
      sigTmp  :*= sig
      tmp :/= sigTmp
      // sigmaTmp := sigma
      // sigmaTmp *= Real.one - Real.one / x.cols
      sigTmp   := -Real.one
      sigTmpB :/= counts
      sigTmp   += Real.one
      sigTmp  :*= sig
      tmp += sigTmp
      */
      /*
      // TODO: Could code this more elegant.
      // 1st term.
      val tmp0 = sig(::, *) :* gradientFactor._1
      // 2nd term.
      val tmp1 = raw.values - mu
      tmp1 :*= tmp1
      tmp1 :/= sig
      tmp1(::, *) :*= gradientFactor._2
      /*
      val w = windowFn(kernel.localIndexOfCenterPair)
      // 1st term.
      val tmp0: DMat = DMat.fill(err.rows, err.cols, -w)
      tmp0(::, *) :/= wSums
      tmp0 += Real.one
      tmp0 :*= sigma
      // 2nd term.
      val tmp1: DMat = raw.values - mu
      tmp1 :*= tmp1
      val tmp1B = tmp1(::, *)
      tmp1B :*= counts
      tmp1 :*= w
      tmp1 :/= sigma
      tmp1B :/= counts
      tmp1B :/= wSums*/
      // Combine
      tmp0 -= tmp1
      tmp0 :/= sig
      tmp0 :/= sig
       */
      case _ =>
        throw new MatchError(context)
    }

    RealArrayTensor.derive(error.layout.size, err)
  }

}

final class LocalContrastNormalizationBuilder
  extends LayerBuilder[LocalContrastNormalizationBuilder]
    with NonTrainableLayerBuilder[LocalContrastNormalizationBuilder] {

  override def repr
  : LocalContrastNormalizationBuilder = this

  private var _kernel
  : Kernel = Kernel2.centered(3, 3)

  def kernel
  : Kernel = _kernel

  def kernel_=(value: Kernel)
  : Unit = {
    require(value.isCentered && value.hasUnitStride)
    _kernel = value
  }

  def setKernel(value: Kernel)
  : LocalContrastNormalizationBuilder = {
    kernel_=(value)
    this
  }

  var window: Window = _

  def setWindow(value: Window)
  : LocalContrastNormalizationBuilder = {
    window = value
    this
  }

  private var _epsilon
  : Real = 1.00000003e-5f

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : LocalContrastNormalizationBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _kernel :: window :: f"${_epsilon}%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LocalContrastNormalizationBuilder]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _kernel.hashCode())
    tmp = MurmurHash3.mix(tmp, window.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LocalContrastNormalizationBuilder =>
      _kernel  == other._kernel &&
      window   == other.window  &&
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : LocalContrastNormalizationBuilder = LocalContrastNormalizationBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: LocalContrastNormalizationBuilder =>
        other._kernel  = _kernel
        other.window   = window
        other._epsilon = _epsilon
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(JVM)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LocalContrastNormalization = new LocalContrastNormalization(
    this, hints, seed, weightsBuilder
  )

}

object LocalContrastNormalizationBuilder {

  final def apply()
  : LocalContrastNormalizationBuilder = new LocalContrastNormalizationBuilder

  final def apply(kernel: Kernel)
  : LocalContrastNormalizationBuilder = apply().setKernel(kernel)

  final def apply(kernel: Kernel, window: Window)
  : LocalContrastNormalizationBuilder = apply(kernel).setWindow(window)

  final def apply(kernel: Kernel, window: Window, epsilon: Real)
  : LocalContrastNormalizationBuilder = {
    apply(kernel, window).setEpsilon(epsilon)
  }

}
/*
final case class LocalContrastNormalizationMetaDataS(mu:    DVec,
                                                     sigma: DVec,
                                                     wCnt:  DenseVector[Int],
                                                     wSum:  DVec)
*/
final case class LocalContrastNormalizationContext(mu:    DenseMatrix[Real],
                                                   sigma: DenseMatrix[Real],
                                                   wCnt:  DenseVector[Int],
                                                   wSum:  DenseVector[Real])
  extends PredictContext {
}


/*
// 50. Single Channel, Kernel with window (optimized implementation TBD)
// 60. Multichannel, Kernel with window
final class NormalizationWithKernelEx(desc:                     NormalizationWithKernelExDesc,
                                  override val bindContext: BindContext,
                                  val          noChannels:  Int,
                                  val          kernel:      KernelLike,
                                  val          window:      WeightingWindowLike,
                                  val          epsilon:     Real)
  extends NormalizationWithKernelLike(desc) {



  override def gradient(mode: Mode, err: DVec, out: DSampleAct, raw: DSampleAct)
  : Unit = err :*= (out.tags.get(id) match {
    case Some(NormalizationWithKernelTagForSample(mu, sig)) =>

    case _ =>
      throw new UnsupportedOperationException
  })

  override def gradient(mode: Mode, err: DMat, out: DBatchAct, raw: DBatchAct)
  : Unit = err :*= (out.tags.get(id) match {
    case Some(NormalizationWithKernelTagForBatch(mu, sig)) =>

    case _ =>
      throw new UnsupportedOperationException
  })

}

final class NormalizationWithKernelExDesc(val noChannels: Int,
                                      val kernel:     KernelLike,
                                      val window:     WeightingWindowLike,
                                      val epsilon:    Real)
  extends LayerFunctionDesc {
  debug_req(kernel.inputSize == kernel.outputSize)
  debug_req(window.noWeights == kernel.size)

  override def toStringEx
  : String = "NormalizeWithKernelEx[%d, %s, %s, %.4g]".format(
    noChannels, kernel, window, epsilon
  )

  override def supports(size: Int)
  : Boolean = size == noChannels * kernel.inputSize

  override def bind(context: BindContext)
  : LayerFunction = new NormalizationWithKernelEx(
    this, context, noChannels, kernel, window, epsilon
  )

}

object NormalizationWithKernelExDesc extends NormalizationDescLike {

  final def apply(noChannels: Int,
                  kernel:     KernelLike,

                  epsilon:    Real = defaultEpsilon)
  : NormalizationWithKernelExDesc = new NormalizationWithKernelExDesc(
    noChannels, kernel, window, epsilon
  )

}

object NormalizationDescLike {

  final val defaultRegularization: Real = 0.01f

}

*/