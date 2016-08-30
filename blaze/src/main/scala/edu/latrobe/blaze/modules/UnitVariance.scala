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

package edu.latrobe.blaze.modules

import java.util.UUID

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.{parameters => par}
import edu.latrobe.sizes._

import scala.collection.Map
import scala.util.hashing._

/**
  * This layer presumes that the input has already zero mean!
  *
  * ---
  * \       !
  * /   x_i = 0
  * ---
  *  i
  *
  *           x_a
  * f(x_a) = -----
  *          sigma
  *
  *       -1
  * f(y_a)   = y_a * sigma
  *
  *                 ---
  *      2      1   \      2
  * sigma  =  ----- /   x_i
  *           m - 1 ---
  *                  i
  *
  *        2          ---       2
  * d sigma      1    \    d x_i
  * -------- = -----  /    ------
  *  d x_a     m - 1  ---  d x_a
  *                    i
  *
  *            2 x_a
  *          = -----
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
  *          = ----- /   x_i di
  *            m - 1 ---
  *                   i
  *
  *         (    /----------- )
  * sigma = (   /      2      )
  *         ( \/  sigma   + e )
  *
  * The epsilon avoids divide by zero and can help to make this function smooth.
  *
  *                            2
  * d sigma      1      d sigma
  * ------- = ------- * --------
  *  d x_a    2 sigma    d x_a
  *
  *              1      2 x_a
  *         = ------- * -----
  *           2 sigma   m - 1
  *
  *                x_a
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
  *         = ------------- /   x_i di
  *           sigma (m - 1) ---
  *                          i
  *
  *            d x_a             d sigma
  *            ----- sigma - x_a -------
  * d f(x_a)   d x_a              d x_a
  * -------- = -------------------------
  *  d x_a                  2
  *                    sigma
  *
  *                        d sigma
  *            sigma - x_a -------
  *                         d x_a
  *          = -------------------
  *                       2
  *                  sigma
  *
  *            d x_a             d sigma
  *            ----- sigma - x_a -------
  * d f(x_a)   d x_b              d x_b
  * -------- = -------------------------
  *  d x_a                  2
  *                    sigma
  *
  *                 d sigma
  *            -x_a -------
  *                  d x_b
  *          = ------------
  *                    2
  *               sigma
  *
  *            ---
  * D f(x_a)   \   d f(x_a)
  * -------- = /   -------- di
  *  D x_a     ---   d x_i
  *             i
  *
  *                        d sigma                d sigma
  *            sigma - x_a -------      ---- -x_a -------
  *                         d x_a       \          d x_i
  *          = ------------------- da + /    ------------ di
  *                       2             ----         2
  *                  sigma              i!=a    sigma
  *
  *                                 d sigma
  *                            ---  -------
  *            sigma           \     d x_i
  *          = ------ da - x_a /   --------- di
  *                 2          ---        2
  *            sigma            i    sigma
  *
  *                           ---
  *                           \   d sigma
  *            sigma da - x_a /   ------- di
  *                           ---  d x_i
  *                            i
  *          = -----------------------------
  *                           2
  *                      sigma
  *
  *                           ---
  *                           \        x_i
  *            sigma da - x_a /   ------------- di
  *                           --- sigma (m - 1)
  *                            i
  *          = -----------------------------------
  *                              2
  *                         sigma
  *
  *                           ---
  *                           \        x_i
  *            sigma da - x_a /   ------------- di      1
  *                           --- sigma (m - 1)       -----
  *                            i                      sigma
  *          = ----------------------------------- * -------
  *                              2                      1
  *                         sigma                     -----
  *                                                   sigma
  *
  *                               ---
  *                      x_a      \
  *            da - ------------- /   x_i di
  *                      2        ---
  *                 sigma (m - 1)  i
  *          = -----------------------------
  *                       sigma
  *
  */
abstract class UnitVariance
  extends MapLayer[UnitVarianceBuilder]
    with NonTrainableLayerLike[UnitVarianceBuilder]
    with NonPenalizing {

  final val domain
  : TensorDomain = builder.domain

  final val learningRate
  : Parameter = builder.learningRate.build("UVLR", seed)

  final val epsilon
  : Real = builder.epsilon

  final val noChannels
  : Int = inputSizeHint.noChannels

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
    runningVarianceReference.foreach(
      initializer(this, _, runningVariance, inputFanSize, outputFanSize)
    )
  }

  // ---------------------------------------------------------------------------
  //    Statistics
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons
  : Long = runningVariance.layout.noValues


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override def extractWeightsFor(neuronNo: Long)
  : Array[Real] = {
    require(neuronNo < runningVariance.layout.noValues)
    Array(runningVariance.get(neuronNo.toInt))
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

  // Running mean and and variance is not a trainable property.
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
  : UnitVarianceState = UnitVarianceState(super.state, learningRate.state)

  final override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: UnitVarianceState =>
        learningRate.restoreState(state.learningRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class UnitVarianceBuilder
  extends MapLayerBuilder[UnitVarianceBuilder] {

  override def repr
  : UnitVarianceBuilder = this

  private var _domain
  : TensorDomain = TensorDomain.Channel

  def domain
  : TensorDomain = _domain

  def domain_=(value: TensorDomain)
  : Unit = {
    require(value != null)
    _domain = value
  }

  def setDomain(value: TensorDomain)
  : UnitVarianceBuilder = {
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
  : UnitVarianceBuilder = {
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
  : UnitVarianceBuilder = {
    epsilon_=(value)
    this
  }

  private var _runningVarianceReference
  : LabeledBufferReference = LabeledBufferReference(1000, 0, "runningVariance")

  def runningVarianceDevReference
  : LabeledBufferReference = _runningVarianceReference

  def runningVarianceDevReference_=(value: LabeledBufferReference)
  : Unit = {
    require(value != null)
    _runningVarianceReference = value
  }

  def setRunningVarianceReference(value: LabeledBufferReference)
  : UnitVarianceBuilder = {
    runningVarianceDevReference_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _domain :: _learningRate :: f"${_epsilon}%.4g" :: _runningVarianceReference :: super.doToString()
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _domain.hashCode())
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp = MurmurHash3.mix(tmp, _runningVarianceReference.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[UnitVarianceBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: UnitVarianceBuilder =>
      _domain                    == other._domain        &&
      _learningRate             == other._learningRate &&
      _epsilon                  == other._epsilon      &&
      _runningVarianceReference == other._runningVarianceReference
    case _ =>
      false
  })

  override protected def doCopy()
  : UnitVarianceBuilder = UnitVarianceBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: UnitVarianceBuilder =>
        other._domain                    = _domain
        other._learningRate             = _learningRate
        other._epsilon                  = _epsilon
        other._runningVarianceReference = _runningVarianceReference
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
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
    if (_runningVarianceReference.segmentNo == 0 || !builder.contains(_runningVarianceReference)) {
      val layout = runningVarianceLayoutFor(hints.layout)
      builder.register(_runningVarianceReference, layout)
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = UnitVarianceBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = UnitVarianceBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {
    runningVarianceDevReference_=(fn(_runningVarianceReference))
  }

}

object UnitVarianceBuilder
  extends ModuleVariantTable[UnitVarianceBuilder] {

  register(2, UnitVariance_JVM_Baseline_Description)

  final def apply()
  : UnitVarianceBuilder = new UnitVarianceBuilder

  final def apply(domain: TensorDomain)
  : UnitVarianceBuilder = apply().setDomain(domain)

  final def apply(domain:       TensorDomain,
                  learningRate: ParameterBuilder)
  : UnitVarianceBuilder = apply(
    domain
  ).setLearningRate(learningRate)

  final def apply(domain:       TensorDomain,
                  learningRate: ParameterBuilder,
                  epsilon:      Real)
  : UnitVarianceBuilder = apply(
    domain,
    learningRate
  ).setEpsilon(epsilon)

}

final case class UnitVarianceState(override val parent: InstanceState,
                                   learningRate:        InstanceState)
  extends ModuleState {
}
