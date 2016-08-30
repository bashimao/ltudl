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

import breeze.linalg.DenseMatrix
import breeze.numerics._
import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
 * A normalization method used by AlexNet. Description can be found in:
 * "ImageNet Classification with Deep Convolutional Neural Networks"
 * (A. Krizhevsky et. al.)
 *
 * i = kernel map index
 * x,y = position in kernel.
 *
 *                        i
 *                       a
 *  ( i  )                x,y
 * f(a   ) = --------------------------------
 *  ( x,y)                               beta
 *           (           min(N-1,i+n/2) )
 *           (           -------------- )
 *           (           \            2 )
 *           ( k + alpha  \   (  j   )  )
 *           (            /   ( a    )  )
 *           (           /    (  x,y )  )
 *           (           -------------- )
 *           (           j=max(0,i-n/2) )
 *
 *                                              -4
 * Krizhevsky suggests: k = 2, n = 5, alpha = 10  , beta = 0.75
 *
 *
 * Derivative (for the sake of brevity, I will omit the x,y subscript and
 * instead put the superscript (i/j) as a inline subscript. Please note that the
 * sum will always include a_i as long as n >= 1. Hence, I will omit this
 * boundary condition here as well. With these rules in place the above equation
 * becomes this.)
 *
 *           a_i                 a_i
 * f(a_i) = ------ = -------------------------
 *          g(a_i)                        beta
 *                   (           ----    )
 *                   (           \     2 )
 *                   ( k + alpha /  a_j  )
 *                   (           ----    )
 *                   (             j     )
 *
 *                         d g(a_i)
 *            g(a_i) - a_i --------
 * d f(a_i)                 d a_i
 * -------- = ---------------------
 *  d a_i                2
 *                 g(a_i)
 *
 *                    ----
 *                    \     2
 * h(a_i) = k + alpha /  a_j
 *                    ----
 *                     j
 *
 * d g(a_i)         beta   (              beta - 1 ) d h(a_i)
 * -------- = h(a_i)     = ( beta * h(a_i)         ) --------
 *  d a_i                  (                       )   d a_i
 *
 * d h(a_i)
 * -------- = 2 alpha a_i
 *  d a_i
 *
 * Hence:
 *
 *                                 beta                                                  beta - 1
 *            (           ----    )                                 (           ----    )
 *            (           \     2 )                             2   (           \     2 )
 *            ( k + alpha /  a_j  )     - 2 * alpha * beta * a_i  * ( k + alpha /  a_j  )
 *            (           ----    )                                 (           ----    )
 * d f(a_i)   (             j     )                                 (             j     )
 * -------- = ---------------------------------------------------------------------------
 *  d a_i                                                2 beta
 *                                  (           ----    )
 *                                  (           \     2 )
 *                                  ( k + alpha /  a_j  )
 *                                  (           ----    )
 *                                  (             j     )
 *
 */
// TODO: Create memory conserving variant!
final class LateralResponseNormalization(override val builder:        LateralResponseNormalizationBuilder,
                                         override val inputHints:     BuildHints,
                                         override val seed:           InstanceSeed,
                                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[LateralResponseNormalizationBuilder]
    with NonTrainableLayer[LateralResponseNormalizationBuilder]
    with NonPenalizing {

  override val outputHints
  : BuildHints = inputHints.derive(JVM)

  val n = builder.n

  val nHalf: Int = n / 2

  val k = builder.k

  val alpha = builder.alpha

  val beta = builder.beta

  private val betaMul2: Real = beta + beta

  private val betaSub1: Real = beta - Real.one

  // TODO: This could be done faster!
  private def computeInnerSum(alphaSqr: DenseMatrix[Real],
                              res:      DenseMatrix[Real])
  : Unit = {
    val NSub1 = alphaSqr.rows - 1
    MatrixEx.foreachRowVectorPair(res)((i, resRow) => {
      val j0           = Math.max(i - nHalf, 0)
      val j1           = Math.min(i + nHalf, NSub1)
      val alphaSqrRows = alphaSqr(j0 until j1, ::)
      MatrixEx.foreachRowVector(alphaSqrRows)(
        alphaSqrRow => resRow += alphaSqrRow
      )
    })
  }

  /*
  override protected def doPredict(mode: ComputeMode, input: SampleTensor)
  : (SampleTensor, Any) = {
    // Compute scaled squares.
    val alphaSqr = input.values :* input.values
    alphaSqr :*= alpha

    // Create lateral view and compute sums.
    val sum = {
      val sqr = alphaSqr.asMatrix(input.size.noChannels)
      val sum = DMat.fill(sqr.rows, sqr.cols, k)
      computeInnerSum(sqr, sum)
      sum.asVector
    }

    // Compute sum^beta and out.
    val sumPowBeta = pow(sum, beta)
    val out        = input.values :/ sumPowBeta

    // Reverse shape of output and return.
    val ctx = LateralResponseNormalizationMetaDataS(alphaSqr, sum, sumPowBeta)
    DenseSampleTensor(out, input.size) -> ctx
  }*/

  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val inpSize = input.layout.size
    // Compute scaled squares.
    val alphaSqr = input.valuesMatrix :* input.valuesMatrix
    alphaSqr :*= alpha

    // Create lateral view and compute sums.
    val sum = {
      val sqr = MatrixEx.reshape(alphaSqr, inpSize.noChannels)
      val sum = MatrixEx.fill(sqr.rows, sqr.cols, k)
      computeInnerSum(sqr, sum)
      MatrixEx.reshape(sum, alphaSqr.rows, alphaSqr.cols)
    }

    // Compute sum^beta and out.
    val sumPowBeta = pow(sum, beta)
    val out        = input.valuesMatrix :/ sumPowBeta

    // Reverse shape of output and return.
    val ctx = LateralResponseNormalizationContext(alphaSqr, sum, sumPowBeta)
    (RealArrayTensor.derive(inpSize, out), ctx)
  }

  /*
  override protected def doPredictInv(mode:    ComputeMode,
                                      output:  SampleTensor,
                                      context: Any)
  : SampleTensor = context match {
    case LateralResponseNormalizationMetaDataS(alphaSqr, sum, sumPowBeta) =>
      DenseSampleTensor(output.values :* sumPowBeta, output.size)
  }*/

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = context match {
    case LateralResponseNormalizationContext(alphaSqr, sum, sumPowBeta) =>
      val outSize = output.layout.size
      RealArrayTensor.derive(outSize, output.valuesMatrix :* sumPowBeta)
    case _ =>
      throw new MatchError(context)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  /*
  override def deriveInputError(mode:      ComputeMode,
                                input:     SampleTensor,
                                output:    SampleTensor,
                                context:   Any,
                                error:     SampleTensor,
                                reference: SampleTensor)
  : SampleTensor = {
    val err = error.values

    context match {
      case LateralResponseNormalizationMetaDataS(alphaSqr, sum, sumPowBeta) =>
        err.transformEx(alphaSqr, sum, sumPowBeta)(
          (err, alphaSqr, sum, sumPowBeta) => {
            val tmp = sumPowBeta - Math.pow(sum, betaSub1) * betaMul2 * alphaSqr
            Real(err * tmp / (sumPowBeta * sumPowBeta))
          }
        )

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
      case LateralResponseNormalizationContext(alphaSqr, sum, sumPowBeta) =>
        MatrixEx.transform(err, alphaSqr, sum, sumPowBeta)(
          (err, alphaSqr, sum, sumPowBeta) => {
            val tmp = sumPowBeta - Math.pow(sum, betaSub1) * betaMul2 * alphaSqr
            Real(err * tmp / (sumPowBeta * sumPowBeta))
          }
        )
      case _ =>
        throw new MatchError(context)
    }

    RealArrayTensor.derive(errSize, err)
  }

}

final class LateralResponseNormalizationBuilder()
  extends LayerBuilder[LateralResponseNormalizationBuilder]
    with NonTrainableLayerBuilder[LateralResponseNormalizationBuilder] {

  override def repr: LateralResponseNormalizationBuilder = this

  private var _n: Int = 5

  def n: Int = _n

  def n_=(value: Int): Unit = {
    require(value >= 1)
    _n = value
  }

  def setN(value: Int): LateralResponseNormalizationBuilder = {
    n_=(value)
    this
  }

  private var _k: Real = 2.0f

  def k: Real = _k

  def k_=(value: Real): Unit = {
    require(value >= Real.zero)
    _k = value
  }

  def setK(value: Real): LateralResponseNormalizationBuilder = {
    k_=(value)
    this
  }

  private var _alpha: Real = 1e-4f

  def alpha: Real = _alpha

  def alpha_=(value: Real): Unit = {
    require(value >= Real.zero)
    _alpha = value
  }

  def setAlpha(value: Real): LateralResponseNormalizationBuilder = {
    alpha_=(value)
    this
  }

  private var _beta: Real = 0.75f

  def beta: Real = _beta

  def beta_=(value: Real): Unit = {
    require(value >= Real.zero)
    _beta = value
  }

  def setBeta(value: Real): LateralResponseNormalizationBuilder = {
    beta_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _n :: f"${_k}%.4g" :: f"${_alpha}%.4g" :: f"${_beta}%.4g" :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _n.hashCode())
    tmp = MurmurHash3.mix(tmp, _k.hashCode())
    tmp = MurmurHash3.mix(tmp, _alpha.hashCode())
    tmp = MurmurHash3.mix(tmp, _beta.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LateralResponseNormalizationBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LateralResponseNormalizationBuilder =>
      _n     == other._n     &&
      _k     == other._k     &&
      _alpha == other._alpha &&
      _beta  == other._beta
    case _ =>
      false
  })

  override protected def doCopy()
  : LateralResponseNormalizationBuilder = LateralResponseNormalizationBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: LateralResponseNormalizationBuilder =>
        other._n     = _n
        other._k     = _k
        other._alpha = _alpha
        other._beta  = _beta
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(JVM)


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LateralResponseNormalization = new LateralResponseNormalization(
    this, hints, seed, weightsBuilder
  )

}

object LateralResponseNormalizationBuilder {

  final def apply()
  : LateralResponseNormalizationBuilder = new LateralResponseNormalizationBuilder

  final def apply(n: Int, k: Real, alpha: Real, beta: Real)
  : LateralResponseNormalizationBuilder = {
    apply().setN(n).setK(k).setAlpha(alpha).setBeta(beta)
  }

}

final case class LateralResponseNormalizationContext(alphaSqr:   DenseMatrix[Real],
                                                     sum:        DenseMatrix[Real],
                                                     sumPowBeta: DenseMatrix[Real])
  extends PredictContext {
}
