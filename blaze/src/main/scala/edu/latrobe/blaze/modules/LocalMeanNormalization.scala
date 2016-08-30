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

import breeze.linalg.{*, DenseMatrix, DenseVector}
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.kernels._
import scala.util.hashing._

/**
 * Performs localized mean normalization using a kernel.
 * (optional: Can make use of compatible weighting windows)
 */
// TODO: Optimized pass for single channel / low channel count!
// TODO: Add code that allows lateral normalization.
final class LocalMeanNormalization(override val builder:        LocalMeanNormalizationBuilder,
                                   override val inputHints:     BuildHints,
                                   override val seed:           InstanceSeed,
                                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[LocalMeanNormalizationBuilder]
    with NonTrainableLayer[LocalMeanNormalizationBuilder]
    with NonPenalizing {

  override val outputHints
  : BuildHints = inputHints.derive(JVM)

  val kernel
  : Kernel = builder.kernel

  val window
  : Window = builder.window.map(window => {
    require(window.noWeights == kernel.noValues)
    window
  }).orNull

  /*
  override protected def doPredict(mode: ComputeMode, input: SampleTensor)
  : (SampleTensor, Any) = {
    val inpSize     = input.size
    val inp         = input.values
    // TODO: Can optimize "w"!
    val derivatives = Array.ofDim[Real](inp.length / input.size.noChannels)
    val mu          = DVec.zeros(inp.length)

    // Incrementally compute mu.
    var i = 0
    if (windowFn == null) {
      kernel.foreachValidPairEx(inpSize, inpSize.noChannels,
        (i0, i1, offset0) => {
          /*val muSlice = {
            val dst0 = i * noChannels
            mu(dst0 until dst0 + noChannels)
          }*/
          val muSlice = mu(i0 until i1)
          var wSum = 0
          Tuple2(
            (j0, j1, offset0, offset1) => {
              /*val src = {
                val src0 = offset * noChannels
                inp(src0 until src0 + noChannels)
              }*/
              val src = inp(offset0 until offset1)
              wSum += 1
              // TODO: Possible performance issue! Int => Float
              lerp.inPlace(muSlice, src, Real.one / wSum)
            },
            () => {
              derivatives(i) = wSum
              i += 1
            }
          )
        }
      )
    }
    else {
      kernel.foreachValidPairIndexEx(inpSize, inpSize.noChannels,
        (i0, i1, offset0) => {
          /*val muSlice = {
            val dst0 = i * noChannels
            mu(dst0 until dst0 + noChannels)
          }*/
          val muSlice = mu(i0 until i1)
          var nSum = 0
          var wSum = Real.zero
          Tuple2(
            (j, offset0, offset1) => {
              /*val src = {
                val src0 = offset * noChannels
                inp(src0 until src0 + noChannels)
              }*/
              val src = inp(offset0 until offset1)
              val w = windowFn(j)
              wSum += w
              nSum += 1
              // TODO: Should add a epsilon here to avoid divide by zero!
              lerp.inPlace(muSlice, src, w / wSum)
            },
            () => {
              derivatives(i) = wSum
              i += 1
            }
          )
        }
      )
    }

    // Compute constants to speed up computation of derivatives.
    if (windowFn == null) {
      derivatives.transform(Real.one - Real.one / _)
    }
    else {
      // TODO: Is this correct?
      val w = windowFn(kernel.localPairNoOfCenterPair)
      derivatives.transform(Real.one - w / _)
    }

    // Remove mean and return.
    val ctx = LocalMeanNormalizationMetaDataS(mu, derivatives)
    DenseSampleTensor(inp - mu, inpSize) -> ctx
  }
  */

  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val inpSize     = input.layout.size
    val inp         = input.valuesMatrix
    // TODO: Can optimize "w"!
    val derivatives = new Array[Real](inp.rows / input.layout.size.noChannels)
    val mu          = DenseMatrix.zeros[Real](inp.rows, inp.cols)

    // Incrementally compute mu.
    var i = 0
    if (window == null) {
      kernel.foreachValidPairEx(inpSize, inpSize.noChannels,
        (i0, i1, offset0) => {
          val muSlice = mu(i0 until i1, ::)
          var wSum = 0
          Tuple2(
            (j0, j1, offset0, offset1) => {
              val src = inp(offset0 until offset1, ::)
              wSum += 1
              // TODO: Possible performance issue! Int => Float
              MatrixEx.lerp(muSlice, src, Real.one / wSum)
            },
            () => {
              derivatives(i) = Real.one - Real.one / wSum
              i += 1
            }
          )
        }
      )
    }
    else {
      val wCenter = window(kernel.localPairNoOfCenterPair)
      kernel.foreachValidPairIndexEx(inpSize, inpSize.noChannels,
        (i0, i1, offset0) => {
          val muSlice = mu(i0 until i1, ::)
          var wSum = Real.zero
          Tuple2(
            (j, offset0, offset1) => {
              val src = inp(offset0 until offset1, ::)
              val w = window(j)
              wSum += w
              // TODO: Should add a epsilon here to avoid divide by zero!
              MatrixEx.lerp(muSlice, src, w / wSum)
            },
            () => {
              derivatives(i) = Real.one - wCenter / wSum
              i += 1
            }
          )
        }
      )
    }

    // Remove mean and return.
    val ctx = LocalMeanNormalizationContext(mu, derivatives)
    (RealArrayTensor.derive(inpSize, inp - mu), ctx)
  }

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = context match {
    case LocalMeanNormalizationContext(mu, derivatives) =>
      RealArrayTensor.derive(output.layout.size, output.valuesMatrix + mu)
  }

  /*
  private lazy val deriveInputErrorFactor: DVec = {
    val tmp = DVec.zeros(kernel.noInputs)
    if (window == null) {
      kernel.foreachValidPair(
        (i, offset) => (j, offset) => tmp.data(i) += Real.one
      )
      tmp.transform(Real.one - Real.one / _)
    }
    else {
      // TODO: Is this correct?
      kernel.foreachValidPair(
        (i, offset) => (j, offset) => tmp.data(i) += window(j)
      )
      val w = window(kernel.localPairIndexOfCenterPair)
      tmp.transform(Real.one - w / _)
    }
    tmp
  }
  */

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  protected def doDeriveInputError(error:       DenseVector[Real],
                                   derivatives: DenseVector[Real],
                                   noChannels:  Int)
  : Unit = {
    val errorValues = VectorEx.asMatrix(error, noChannels)
    errorValues(*, ::) :*= derivatives//deriveInputErrorFactor
  }

  /*
  override def deriveInputError(mode:      ComputeMode,
                                input:     SampleTensor,
                                output:    SampleTensor,
                                context:   Any,
                                error:     SampleTensor,
                                reference: SampleTensor)
  : SampleTensor = {
    val errSize = error.size
    val err     = error.values

    context match {
      case LocalMeanNormalizationMetaDataS(mu, derivatives) =>
        doDeriveInputError(err, DVec(derivatives), errSize.noChannels)

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
    val errSize = error.layout.size
    val err     = error.valuesMatrix

    context match {
      case LocalMeanNormalizationContext(mu, derivatives) =>
        val tmp = DenseVector(derivatives)
        MatrixEx.foreachColumnVector(err)(
          doDeriveInputError(_, tmp, errSize.noChannels)
        )

      case _ =>
        throw new MatchError(context)
    }

    RealArrayTensor.derive(error.layout.size, err)
  }

}

final class LocalMeanNormalizationBuilder
  extends LayerBuilder[LocalMeanNormalizationBuilder]
    with NonTrainableLayerBuilder[LocalMeanNormalizationBuilder] {

  override def repr
  : LocalMeanNormalizationBuilder = this

  private var _kernel
  : Kernel = Kernel2.centered(3, 3)

  def kernel
  : Kernel = _kernel

  def kernel_=(value: Kernel): Unit = {
    require(value.isCentered && value.hasUnitStride)
    _kernel = value
  }

  def setKernel(value: Kernel): LocalMeanNormalizationBuilder = {
    kernel_=(value)
    this
  }

  private var _window
  : Option[Window] = None

  def window
  : Option[Window] = _window

  def window_=(value: Option[Window])
  : Unit = {
    require(value != null)
    _window = value
  }

  def window_=(value: Window)
  : Unit = _window = Option(value)

  def setWindow(value: Option[Window])
  : LocalMeanNormalizationBuilder = {
    window_=(value)
    this
  }

  def setWindow(value: Window)
  : LocalMeanNormalizationBuilder = {
    window_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _kernel :: _window :: super.doToString()

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _kernel.hashCode())
    tmp = MurmurHash3.mix(tmp, _window.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LocalMeanNormalizationBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LocalMeanNormalizationBuilder =>
      _kernel == other._kernel &&
      _window  == other._window
    case _ =>
      false
  })

  override protected def doCopy()
  : LocalMeanNormalizationBuilder = LocalMeanNormalizationBuilder()


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(JVM)


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LocalMeanNormalization = new LocalMeanNormalization(
    this, hints, seed, weightsBuilder
  )

}

object LocalMeanNormalizationBuilder {

  final def apply()
  : LocalMeanNormalizationBuilder = new LocalMeanNormalizationBuilder

  final def apply(kernel: Kernel)
  : LocalMeanNormalizationBuilder = apply().setKernel(kernel)

  final def apply(kernel: Kernel, window: Option[Window])
  : LocalMeanNormalizationBuilder = apply(kernel).setWindow(window)

  final def apply(kernel: Kernel, window: Window)
  : LocalMeanNormalizationBuilder = apply(kernel).setWindow(window)

}

final case class LocalMeanNormalizationContext(mu:          DenseMatrix[Real],
                                               derivatives: Array[Real])
  extends PredictContext {
}
