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

import java.util.UUID

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.generic._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.{parameters => par}
import edu.latrobe.sizes._

import scala.collection.Map
import scala.util.hashing._

/**
  * Applies a basic normalization algorithm that causes the resulting values to
  * have zero mean and unit variance.
  *
  *          x_a - mu
  * f(x_a) = --------
  *           sigma
  *
  *
  * BASIC ALGORITHM
  * ---------------
  *
  *        ---
  *      1 \
  * mu = - /   x_i
  *      m ---
  *         i
  *
  *  d mu   1
  * ----- = -
  * d x_a   m
  *
  *    d mu       1
  * ----------- = -
  * d x_b, a!=b   m
  *
  *         ---              ---
  * D mu    \   d mu       1 \
  * ----- = /   ----- di = - /   di
  * D x_a   --- d x_i      m ---
  *          i                i
  *
  *                 ---
  *      2      1   \             2
  * sigma  =  ----- /   (x_i - mu)
  *           m - 1 ---
  *                  i
  *
  *        2          ---              2
  * d sigma      1    \    d (x_i - mu)
  * -------- = -----  /    -------------
  *  d x_a     m - 1  ---      d x_a
  *                    i
  *
  *                  (                          ----                        )
  *              1   (              (     1 )   \                 (     1 ) )
  *          = ----- ( 2 (x_a - mu) ( 1 - - ) + /    2 (x_i - mu) ( 0 - - ) )
  *            m - 1 (              (     m )   ----              (     m ) )
  *                  (                          i!=a                        )
  *
  *                  (              ---                      )
  *              2   (              \   (     1 )            )
  *          = ----- ( (x_a - mu) + /   ( 0 - - ) (x_i - mu) )
  *            m - 1 (              --- (     m )            )
  *                  (               i                       )
  *
  *                  (                ---            )
  *              2   (              1 \              )
  *          = ----- ( (x_a - mu) - - /   (x_i - mu) )
  *            m - 1 (              m ---            )
  *                  (                 i             )
  *
  *              2
  *          = ----- (x_a - mu)
  *            m - 1
  *
  *        2   ---        2
  * D sigma    \   d sigma
  * -------- = /   -------- di
  *  D x_a     ---  d x_i
  *             i
  *
  *                  ---
  *              2   \
  *          = ----- /   (x_i - mu) di
  *            m - 1 ---
  *                   i
  *
  *         (    /----------- )
  * sigma = (   /      2      )
  *         ( \/  sigma   + e )
  *
  * The epsilon can help to make this function smooth.
  *
  *                            2
  * d sigma      1      d sigma
  * ------- = ------- * --------
  *  d x_a    2 sigma    d x_a
  *
  *               1       2
  *         = ------- * ----- (x_a - mu)
  *           2 sigma   m - 1
  *
  *            (x_a - mu)
  *         = -------------
  *           sigma (m - 1)
  *
  *           ---
  * D sigma   \   d sigma
  * ------- = /   ------- di
  *  D x_a    ---  d x_i
  *            i
  *
  *                         ---
  *                 1       \
  *         = ------------- /   (x_i - mu) di
  *           sigma (m - 1) ---
  *                          i
  *
  *            d x_a             d sigma    d mu            d sigma
  *            ----- sigma - x_a ------- - ----- sigma + mu -------
  * d f(x_a)   d x_a              d x_a    d x_a             d x_a
  * -------- = ----------------------------------------------------
  *  d x_a                               2
  *                                 sigma
  *
  *            (     d mu  )                    d sigma
  *            ( 1 - ----- ) sigma + (mu - x_a) -------
  *            (     d x_a )                     d x_a
  *          = ----------------------------------------
  *                                 2
  *                            sigma
  *
  *               d x_a             d sigma     d mu            d sigma
  *               ----- sigma - x_a ------- -  ----- sigma + mu -------
  *   d f(x_a)    d x_b              d x_b     d x_b              x_b
  * ----------- = -----------------------------------------------------
  * d x_b, b!=a                              2
  *                                     sigma
  *
  *                       d mu               d sigma
  *                -sigma ----- + (mu - x_a) -------
  *                       d x_b               d x_b
  *             = ----------------------------------
  *                                  2
  *                             sigma
  *
  *            ---
  * D f(x_a)   \    d f
  * -------- = /   ----- di
  *  D x_a     --- d x_i
  *             i
  *
  *
  *        (     d mu  )                    d sigma                  d mu               d sigma
  *        ( 1 - ----- ) sigma + (mu - x_a) -------      ---- -sigma ----- + (mu - x_a) -------
  *        (     d x_a )                     d x_a       \           d x_i               d x_i
  *      = ---------------------------------------- da + /    --------------------------------- di
  *                              2                       ----                    2
  *                         sigma                        i!=a               sigma
  *
  *                                                          ----
  *                                                          \    (         d mu              d sigma )
  *        (               d mu               d sigma )      /    ( -sigma ----- + (mu - x_a) ------- ) di
  *        ( sigma - sigma ----- + (mu - x_a) ------- ) da + ---- (        d x_i               d x_i  )
  *        (               d x_a               d x_a  )      i!=a
  *      = -----------------------------------------------------------------------------------------------
  *                                                      2
  *                                                 sigma
  *
  *                   ---
  *                   \   (         d mu              d sigma )
  *        sigma da + /   ( -sigma ----- + (mu - x_a) ------- ) di
  *                   --- (        d x_i               d x_i  )
  *                    i
  *      = ------------------------------------------------------
  *                                    2
  *                               sigma
  *
  *
  *                         ---                           ---
  *                         \   ( d mu     )              \   ( d sigma    )
  *        sigma da - sigma /   ( ----- di ) + (mu - x_a) /   ( ------- di )
  *                         --- ( d x_i    )              --- (  d x_i     )
  *                          i                             i
  *      = -----------------------------------------------------------------
  *                                         2
  *                                    sigma
  *
  *                         ---                                     ---
  *                         \   ( 1    )                    1       \
  *        sigma da - sigma /   ( - di ) + (mu - x_a) ------------- /   ( (x_i - mu) di )
  *                         --- ( m    )              sigma (m - 1) ---
  *                          i                                       i
  *      = ------------------------------------------------------------------------------
  *                                              2
  *                                         sigma
  *
  *                         ---                          ---
  *                         \   ( 1    )     mu - x_a    \
  *        sigma da - sigma /   ( - di ) + ------------- /   (x_i - mu) di
  *                         --- ( m    )   sigma (m - 1) ---
  *                          i                            i
  *      = ---------------------------------------------------------------
  *                                         2
  *                                    sigma
  *
  *              (        ---    )                 ---
  *              (      1 \      )      mu - x_a   \
  *        sigma ( da - - /   di ) + ------------- /   (x_i - mu) di
  *              (      m ---    )   sigma (m - 1) ---
  *              (         i     )                  i
  *      = ---------------------------------------------------------
  *                                    2
  *                               sigma
  *
  *              (        ---    )                 ---
  *              (      1 \      )      mu - x_a   \                      1
  *        sigma ( da - - /   di ) + ------------- /   (x_i - mu) di    -----
  *              (      m ---    )   sigma (m - 1) ---                  sigma
  *              (         i     )                  i
  *      = --------------------------------------------------------- * -------
  *                                    2                                  1
  *                               sigma                                 -----
  *                                                                     sigma
  *
  *               ---        mu - x_a    ---
  *             1 \        ------------- \
  *        da - - /   di +      2        /   (x_i - mu) di
  *             m ---      sigma (m - 1) ---
  *                i                      i
  *      = -----------------------------------------------
  *                            sigma
  *
  * REMARK: For population statistics replace (m - 1) with m.
  *
  *
  * WITH NON-UNIFORM WINDOW FUNCTION
  * --------------------------------
  *
  *               ---
  *         1     \
  * mu = -------  /   w_i x_i
  *      ---      ---
  *      \         i
  *      /   w_i
  *      ---
  *       i
  *
  *  d mu     w_a
  * ----- = -------
  * d x_a   ---
  *         \
  *         /   w_i
  *         ---
  *          i
  *
  *         ---                    ---
  * D mu    \   d mu          1    \
  * ----- = /   ----- di = ------- /   w_i di
  * D x_a   --- d x_i      ---     ---
  *          i             \        i
  *                        /   w_i
  *                        ---
  *                         i
  *
  * m' = Number of non-zero weights. You can also neutralize m' terms by
  *      injecting 1 if you do not want to stencil out zero weights.
  *
  *                           ---
  *      2          m'        \                 2
  * sigma  = ---------------- /   w_i (x_i - mu)
  *                   ---     ---
  *                   \        i
  *          (m' - 1) /   w_i
  *                   ---
  *                    i
  *
  *        2                    ---                 2
  * d sigma           m'        \       d (x_i - mu)
  * -------- = ---------------- /   w_i -------------
  *  d x_a              ---     ---         d x_a
  *                     \        i
  *            (m' - 1) /   w_i
  *                     ---
  *                      i
  *
  *          = See above for details on how I reduce the terms step by step!
  *
  *                  2 m'
  *          = ---------------- w_a (x_a - mu)
  *                     ---
  *                     \
  *            (m' - 1) /   w_i
  *                     ---
  *                      i
  *
  *        2   ---        2                       ---
  * D sigma    \   d sigma            2 m'        \
  * -------- = /   -------- di = ---------------- /   w_i (x_i - mu) di
  *  D x_a     ---  d x_i                 ---     ---
  *             i                         \        i
  *                              (m' - 1) /   w_i
  *                                       ---
  *                                        i
  *
  *            /------
  * sigma =   /      2
  *         \/  sigma
  *
  *                            2
  * d sigma      1      d sigma
  * ------- = ------- * --------
  *  d x_a    2 sigma    d x_a
  *
  *               1           2 m'
  *         = ------- * ---------------- w_a (x_a - mu)
  *           2 sigma            ---
  *                              \
  *                     (m' - 1) /   w_i
  *                              ---
  *                               i
  *
  *              m' w_a (x_a - mu)
  *         = ----------------------
  *                          ---
  *                          \
  *           sigma (m' - 1) /   w_i
  *                          ---
  *                           i
  *
  *            ---                                     ---
  * D sigma    \   d sigma                m'           \
  * -------- = /   ------- di = ---------------------- /   w_i (x_i - mu) di
  *  D x_a     ---  d x_i                      ---     ---
  *             i                              \        i
  *                             sigma (m' - 1) /   w_i
  *                                            ---
  *                                             i
  *
  *          x_a - mu
  * f(x_a) = --------
  *           sigma
  *
  *  d f(x_a)
  * --------- = See above!
  *   d x_a
  *
  *   d f(x_a)
  * ----------- = See above!
  * d x_b, b!=a
  *
  *         ---
  *  D f    \     d f
  * ----- = /   ------- di
  * D x_a   ---  d x_i
  *          i
  *
  *                     ---                                 ---
  *                1    \                m' (mu - x_a)      \
  *        da - ------- /   w_i di + ---------------------- /   w_i (x_i - mu) di
  *             ---     ---                         ---     ---
  *             \        i                2         \        i
  *             /   w_i              sigma (m' - 1) /   w_i
  *             ---                                 ---
  *              i                                   i
  *      = ----------------------------------------------------------------------
  *                                         sigma
  *
  *
  * NORMALIZED WINDOW WITH TOTAL COVERAGE
  * -------------------------------------
  *
  * ---
  * \
  * /   w_i = 1
  * ---
  *  i
  *
  *      ---
  *      \
  * mu = /   w_i x_i
  *      ---
  *       i
  *
  *  d mu
  * ----- = w_a
  * d x_a
  *
  *         ---            ---
  * D mu    \   d mu       \
  * ----- = /   ----- di = /   w_i di
  * D x_a   --- d x_i      ---
  *          i              i
  *
  *
  * m' = Number of non-zero weights. You can also neutralize m' terms by
  *      injecting 1 if you do not want to stencil out zero weights.
  *
  *      2     m'                 2
  * sigma  = ------ w_i (x_i - mu)
  *          m' - 1
  *
  *        2          ---                 2
  * d sigma      m'   \       d (x_i - mu)
  * -------- = ------ /   w_i -------------
  *  d x_a     m' - 1 ---         d x_a
  *                    i
  *
  *          = See above!
  *
  *             2 m'
  *          = ------ w_a (x_a - mu)
  *            m' - 1
  *
  *        2   ---        2
  * D sigma    \   d sigma
  * -------- = /   -------- di
  *  D x_a     ---  d x_i
  *             i
  *
  *                   ---
  *             2 m'  \
  *          = ------ /   w_i (x_i - mu)
  *            m' - 1 ---
  *                    i
  *
  *            /------
  * sigma =   /      2
  *         \/  sigma
  *
  *                            2
  * d sigma      1      d sigma
  * ------- = ------- * --------
  *  d x_a    2 sigma    d x_a
  *
  *              1       2 m'
  *         = ------- * ------ w_a (x_a - mu)
  *           2 sigma   m' - 1
  *
  *           m' w_a (x_a - mu)
  *         = -----------------
  *            sigma (m' - 1)
  *
  *            ---
  * D sigma    \   d sigma
  * -------- = /   ------- di
  *  D x_a     ---  d x_i
  *             i
  *
  *                           ---
  *                  m'       \
  *          = -------------- /   w_i (x_i - mu)
  *            sigma (m' - 1) ---
  *                            i
  *
  *          x_a - mu
  * f(x_a) = --------
  *           sigma
  *
  *  d f(x_a)
  * --------- = See above!
  *   d x_a
  *
  *   d f(x_a)
  * ----------- = See above!
  * d x_b, b!=a
  *
  *         ---
  *  D f    \     d f
  * ----- = /   ------- di
  * D x_a   ---  d x_i
  *          i
  *
  *             ---                         ---
  *             \            m' (mu - x_a)  \
  *        da - /   w_i di + -------------- /   w_i (x_i - mu) di
  *             ---               2         ---
  *              i           sigma (m' - 1)  i
  *      = ------------------------------------------------------
  *                               sigma
  *
  *
  * During training mode, we try to collect figures on the running mean and
  * std-deviation. In test mode we then use these figures.
  *
  */
abstract class Normalization
  extends MapLayer[NormalizationBuilder]
    with NonTrainableLayerLike[NormalizationBuilder]
    with NonPenalizing {

  final val domain
  : TensorDomain = builder.domain

  final val learningRate
  : Parameter = builder.learningRate.build("NLR", seed)

  final val epsilon
  : Real = builder.epsilon

  final val noChannels
  : Int = inputSizeHint.noChannels

  final val runningMeanLayout
  : IndependentTensorLayout = builder.runningMeanLayoutFor(inputLayoutHint)

  def runningMeanReference
  : Option[LabeledBufferReference]

  def runningMean: ValueTensor

  final val runningVarianceLayout
  : IndependentTensorLayout = builder.runningVarianceLayoutFor(inputLayoutHint)

  def runningVarianceReference
  : Option[LabeledBufferReference]

  def runningVariance
  : ValueTensor

  @transient
  final override lazy val weightReferences
  : Set[LabeledBufferReference] = {
    val builder = Set.newBuilder[LabeledBufferReference]
    runningMeanReference.map(builder += _)
    runningVarianceReference.map(builder += _)
    builder.result()
  }

  final override val parameters
  : Map[UUID, Parameter] = super.parameters + learningRate.toTuple

  override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize = domain match {
      case TensorDomain.Unit    => inputLayoutHint.noSamples
      case TensorDomain.Channel => inputLayoutHint.noTuples
      case TensorDomain.Sample  => inputSizeHint.noValues
      case TensorDomain.Batch   => inputLayoutHint.noValues
    }
    val outputFanSize = domain match {
      case TensorDomain.Unit    => outputHints.layout.noSamples
      case TensorDomain.Channel => outputHints.layout.noTuples
      case TensorDomain.Sample  => outputHints.layout.size.noValues
      case TensorDomain.Batch   => outputHints.layout.noValues
    }
    runningMeanReference.foreach(
      initializer(this, _, runningMean, inputFanSize, outputFanSize)
    )
    runningVarianceReference.foreach(
      initializer(this, _, runningVariance, inputFanSize, outputFanSize)
    )
  }


  // ---------------------------------------------------------------------------
  //    Statistics
  // ---------------------------------------------------------------------------
  final override val noNeurons
  : Long = runningMeanLayout.noValues + runningVarianceLayout.noValues


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override def extractWeightsFor(neuronNo: Long)
  : Array[Real] = {
    var n = neuronNo
    if (n < runningMeanLayout.noValues) {
      return Array(runningMean.get(n.toInt))
    }
    n -= runningMeanLayout.noValues
    if (n < runningVarianceLayout.noValues) {
      return Array(runningVariance.get(n.toInt))
    }
    throw new IndexOutOfBoundsException
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = mode match {
    case mode: Training =>
      // Compute learning rate.
      val lr = learningRate.get(mode.phaseNo, RealRange.zeroToInfinity)
      //learningRate.update(mode.phaseNo, Real.one)

      // Select predict function.
      domain match {
        case TensorDomain.Unit =>
          require(input.layout.size == inputSizeHint)
          doPredictForUnitTraining(input, lr)

        case TensorDomain.Channel =>
          require(input.layout.size.noChannels == inputSizeHint.noChannels)
          doPredictForChannelTraining(input, lr)

        case TensorDomain.Sample =>
          require(input.layout.noSamples == inputLayoutHint.noSamples)
          doPredictForSampleTraining(input, lr)

        case TensorDomain.Batch =>
          doPredictForBatchTraining(input, lr)

        case _ =>
          throw new MatchError(domain)
      }

    case mode: Inference =>
      val out = domain match {
        case TensorDomain.Unit =>
          require(input.layout.size == inputSizeHint)
          doPredictForUnitInference(inPlaceAllowed, input)

        case TensorDomain.Channel =>
          require(input.layout.size.noChannels == inputSizeHint.noChannels)
          doPredictForChannelInference(inPlaceAllowed, input)

        case TensorDomain.Sample =>
          require(input.layout.noSamples == inputLayoutHint.noSamples)
          doPredictForSampleInference(inPlaceAllowed, input)

        case TensorDomain.Batch =>
          doPredictForBatchInference(inPlaceAllowed, input)

        case _ =>
          throw new MatchError(domain)
      }
      (out, EmptyContext)

    case _ =>
      throw new MatchError(mode)
  }

  protected def doPredictForUnitTraining(input:        Tensor,
                                         learningRate: Real)
  : (Tensor, PredictContext)

  protected def doPredictForUnitInference(inPlaceAllowed: Boolean,
                                          input:          Tensor)
  : Tensor

  protected def doPredictForChannelTraining(input:        Tensor,
                                            learningRate: Real)
  : (Tensor, PredictContext)

  protected def doPredictForChannelInference(inPlaceAllowed: Boolean,
                                             input:          Tensor)
  : Tensor

  protected def doPredictForSampleTraining(input:        Tensor,
                                           learningRate: Real)
  : (Tensor, PredictContext)

  protected def doPredictForSampleInference(inPlaceAllowed: Boolean,
                                            input:          Tensor)
  : Tensor

  protected def doPredictForBatchTraining(input:        Tensor,
                                          learningRate: Real)
  : (Tensor, PredictContext)

  protected def doPredictForBatchInference(inPlaceAllowed: Boolean,
                                           input:          Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new NotImplementedError


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  // Running mean and and variance is not a normal trainable property. They
  // are updated directly during the forward pass.
  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = domain match {
    case TensorDomain.Unit =>
      require(error.layout.size == inputSizeHint)
      doDeriveInputErrorForUnit(input, context, error)

    case TensorDomain.Channel =>
      require(error.layout.size.noChannels == inputSizeHint.noChannels)
      doDeriveInputErrorForChannel(input, context, error)

    case TensorDomain.Sample =>
      require(error.layout.noSamples == inputLayoutHint.noSamples)
      doDeriveInputErrorForSample(input, context, error)

    case TensorDomain.Batch =>
      doDeriveInputErrorForBatch(input, context, error)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doDeriveInputErrorForUnit(input:   Tensor,
                                          context: PredictContext,
                                          error:   Tensor)
  : Tensor

  protected def doDeriveInputErrorForChannel(input:   Tensor,
                                             context: PredictContext,
                                             error:   Tensor)
  : Tensor


  protected def doDeriveInputErrorForSample(input:   Tensor,
                                            context: PredictContext,
                                            error:   Tensor)
  : Tensor


  protected def doDeriveInputErrorForBatch(input:   Tensor,
                                           context: PredictContext,
                                           error:   Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //   State management.
  // ---------------------------------------------------------------------------
  final override def state
  : NormalizationState = NormalizationState(
    super.state,
    learningRate.state
  )

  final override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: NormalizationState =>
        learningRate.restoreState(state.learningRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class NormalizationBuilder
  extends MapLayerBuilder[NormalizationBuilder] {

  override def repr
  : NormalizationBuilder = this

  private var _domain
  : TensorDomain = TensorDomain.Batch

  def domain
  : TensorDomain = _domain

  def domain_=(value: TensorDomain)
  : Unit = {
    require(value != null)
    _domain = value
  }

  def setDomain(value: TensorDomain)
  : NormalizationBuilder = {
    domain_=(value)
    this
  }

  private var _learningRate
  : ParameterBuilder = par.CMAFactorBuilder().clip(
    0.01f, Real.positiveInfinity
  )

  def learningRate
  : ParameterBuilder = _learningRate

  def learningRate_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _learningRate = value
  }

  def setLearningRate(value: ParameterBuilder)
  : NormalizationBuilder = {
    learningRate_=(value)
    this
  }

  /**
    * Constant epsilon to avoid divide by zero.
    */
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
  : NormalizationBuilder = {
    epsilon_=(value)
    this
  }

  private var _runningMeanReference
  : LabeledBufferReference = LabeledBufferReference(1000, 0, "runningMean")

  def runningMeanReference
  : LabeledBufferReference = _runningMeanReference

  def runningMeanReference_=(value: LabeledBufferReference)
  : Unit = {
    require(value != null)
    _runningMeanReference = value
  }

  def setRunningMeanReference(value: LabeledBufferReference)
  : NormalizationBuilder = {
    runningMeanReference_=(value)
    this
  }

  private var _runningVarianceReference
  : LabeledBufferReference = LabeledBufferReference(1000, 0, "runningVariance")

  def runningVarianceReference
  : LabeledBufferReference = _runningVarianceReference

  def runningVarianceReference_=(value: LabeledBufferReference)
  : Unit = {
    require(value != null)
    _runningVarianceReference = value
  }

  def setRunningVarianceReference(value: LabeledBufferReference)
  : NormalizationBuilder = {
    runningVarianceReference_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _domain :: _learningRate :: f"${_epsilon}%.4g" :: _runningMeanReference :: _runningVarianceReference :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _domain.hashCode())
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp = MurmurHash3.mix(tmp, _runningMeanReference.hashCode())
    tmp = MurmurHash3.mix(tmp, _runningVarianceReference.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[NormalizationBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: NormalizationBuilder =>
      _domain                   == other._domain               &&
      _learningRate             == other._learningRate         &&
      _epsilon                  == other._epsilon              &&
      _runningMeanReference     == other._runningMeanReference &&
      _runningVarianceReference == other._runningVarianceReference
    case _ =>
      false
  })

  override protected def doCopy()
  : NormalizationBuilder = NormalizationBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: NormalizationBuilder =>
        other._domain                    = _domain
        other._learningRate             = _learningRate
        other._epsilon                  = _epsilon
        other._runningMeanReference     = _runningMeanReference
        other._runningVarianceReference = _runningVarianceReference
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  def runningMeanLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = _domain match {
    case TensorDomain.Unit =>
      layoutHint.derive(1)
    case TensorDomain.Channel =>
      IndependentTensorLayout.derive(layoutHint.size.noChannels, 1)
    case TensorDomain.Sample =>
      layoutHint.derive(Size1.one)
    case TensorDomain.Batch =>
      IndependentTensorLayout.one
    case _ =>
      throw new MatchError(_domain)
  }

  def runningVarianceLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = _domain match {
    case TensorDomain.Unit =>
      layoutHint.derive(1)
    case TensorDomain.Channel =>
      IndependentTensorLayout.derive(layoutHint.size.noChannels, 1)
    case TensorDomain.Sample =>
      layoutHint.derive(Size1.one)
    case TensorDomain.Batch =>
      IndependentTensorLayout.one
    case _ =>
      throw new MatchError(_domain)
  }

  override protected def doWeightLayoutFor(hints:   BuildHints,
                                           builder: TensorLayoutBufferBuilder)
  : Unit = {
    if (_runningMeanReference.segmentNo == 0 || !builder.contains(_runningMeanReference)) {
      val layout = runningMeanLayoutFor(hints.layout)
      builder.register(_runningMeanReference, layout)
    }
    if (_runningVarianceReference.segmentNo == 0 || !builder.contains(_runningVarianceReference)) {
      val layout = runningVarianceLayoutFor(hints.layout)
      builder.register(_runningVarianceReference, layout)
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = NormalizationBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = NormalizationBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {
    runningMeanReference_=(fn(_runningMeanReference))
    runningVarianceReference_=(fn(_runningVarianceReference))
  }

}

object NormalizationBuilder
  extends ModuleVariantTable[NormalizationBuilder] {

  register(2, Normalization_JVM_Baseline_Description)
  register(64, Normalization_Generic_Baseline_Description)

  final def apply()
  : NormalizationBuilder = new NormalizationBuilder

  final def apply(domain: TensorDomain)
  : NormalizationBuilder = apply().setDomain(domain)

  final def apply(domain:       TensorDomain,
                  learningRate: ParameterBuilder)
  : NormalizationBuilder = apply(
    domain
  ).setLearningRate(learningRate)

  final def apply(domain:       TensorDomain,
                  learningRate: ParameterBuilder,
                  epsilon:      Real)
  : NormalizationBuilder = apply(
    domain,
    learningRate
  ).setEpsilon(epsilon)

}

final case class NormalizationState(override val parent: InstanceState,
                                    learningRate:        InstanceState)
  extends ModuleState {
}
