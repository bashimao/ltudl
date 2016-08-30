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
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.{parameters => par}
import edu.latrobe.sizes.Size1

import scala.collection.Map
import scala.util.hashing._

/**
  * Makes activations to have a mean value of 0.0.
  *
  * f(x_a) = x_a - mu
  *
  *
  * BASIC ALGORITHM
  * ---------------
  *
  *         m
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
  *         ---              ---
  * D mu    \   d mu       1 \
  * ----- = /   ----- di = - /   di
  * D x_a   --- d x_i      m ---
  *          i                i
  *
  * f(x_a) = x_a - mu
  *
  * d f(x_a)        d mu       1
  * -------- = 1 - ----- = 1 - -
  *  d x_a         d x_a       m
  *
  *   d f(x_a)         d mu       1
  * ----------- = 0 - ----- = 0 - -
  * d x_b, b!=a       d x_a       m
  *
  *         ---
  *  D f    \     d f
  * ----- = /   ------- di
  * D x_a   ---  d x_i
  *          i
  *
  *                            ----
  *         (     d mu  )      \     d mu
  *       = ( 1 - ----- ) da + /    ----- di
  *         (    d x_a  )      ---- d x_i
  *                            i!=a
  *
  *                           ----
  *                 d mu      \     d mu
  *       = 1 da - ----- da + /    ----- di
  *                d x_a      ---- d x_i
  *                           i!=a
  *
  *                       ----
  *                1      \    1
  *       = 1 da - - da - /    - di
  *                m      ---- m
  *                       i!=a
  *
  *                ---
  *              1 \
  *       = da - - /   di
  *              m ---
  *                 i
  *
  *
  * WITH NON-UNIFORM WINDOW FUNCTION
  * --------------------------------
  *
  * f(x_a) = x_a - mu
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
  *                      ---
  *                 1    \
  *       = da - ------- /   w_i di
  *              ---     ---
  *              \
  *              /   w_i
  *              ---
  *               i
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
  * f(x_a) = x_a - mu
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
  *              ---
  *              \
  *       = da - /   w_i di
  *              ---
  *               i
  *
  */
abstract class ZeroMean
  extends MapLayer[ZeroMeanBuilder]
    with NonTrainableLayerLike[ZeroMeanBuilder]
    with NonPenalizing {

  final val domain
  : TensorDomain = builder.domain

  final val learningRate
  : Parameter = builder.learningRate.build("ZMLR", seed)

  final val runningMeanLayout
  : IndependentTensorLayout = builder.runningMeanLayoutFor(inputLayoutHint)

  def runningMeanReference
  : Option[LabeledBufferReference]

  def runningMean
  : ValueTensor

  @transient
  final override lazy val weightReferences
  : Set[LabeledBufferReference] = {
    val builder = Set.newBuilder[LabeledBufferReference]
    runningMeanReference.map(builder += _)
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
  }


  // ---------------------------------------------------------------------------
  //    Statistics
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons
  : Long = runningMean.layout.noValues


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override def extractWeightsFor(neuronNo: Long)
  : Array[Real] = {
    require(neuronNo < runningMean.layout.noValues)
    Array(runningMean.get(neuronNo.toInt))
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
          doPredictForUnitTraining(inPlaceAllowed, input, lr)

        case TensorDomain.Channel =>
          require(input.layout.size.noChannels == inputSizeHint.noChannels)
          doPredictForChannelTraining(inPlaceAllowed, input, lr)

        case TensorDomain.Sample =>
          require(input.layout.noSamples == inputLayoutHint.noSamples)
          doPredictForSampleTraining(inPlaceAllowed, input, lr)

        case TensorDomain.Batch =>
          doPredictForBatchTraining(inPlaceAllowed, input, lr)

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

  protected def doPredictForUnitTraining(inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         learningRate:   Real)
  : (Tensor, PredictContext)

  protected def doPredictForUnitInference(inPlaceAllowed: Boolean,
                                          input:          Tensor)
  : Tensor

  protected def doPredictForChannelTraining(inPlaceAllowed: Boolean,
                                            input:          Tensor,
                                            learningRate:   Real)
  : (Tensor, PredictContext)

  protected def doPredictForChannelInference(inPlaceAllowed: Boolean,
                                             input:          Tensor)
  : Tensor

  protected def doPredictForSampleTraining(inPlaceAllowed: Boolean,
                                           input:          Tensor,
                                           learningRate:   Real)
  : (Tensor, PredictContext)

  protected def doPredictForSampleInference(inPlaceAllowed: Boolean,
                                            input:          Tensor)
  : Tensor

  protected def doPredictForBatchTraining(inPlaceAllowed: Boolean,
                                          input:          Tensor,
                                          learningRate:   Real)
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
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  // Running mean and and stdDev is not a trainable property.
  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = domain match {
    case TensorDomain.Unit =>
      require(error.layout.size == inputSizeHint)
      doDeriveInputErrorForUnit(context, error)

    case TensorDomain.Channel =>
      require(error.layout.size.noChannels == inputSizeHint.noChannels)
      doDeriveInputErrorForChannel(context, error)

    case TensorDomain.Sample =>
      require(error.layout.noSamples == inputLayoutHint.noSamples)
      doDeriveInputErrorForSample(context, error)

    case TensorDomain.Batch =>
      doDeriveInputErrorForBatch(context, error)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doDeriveInputErrorForUnit(context: PredictContext,
                                          error:   Tensor)
  : Tensor

  protected def doDeriveInputErrorForChannel(context: PredictContext,
                                             error:   Tensor)
  : Tensor


  protected def doDeriveInputErrorForSample(context: PredictContext,
                                            error:   Tensor)
  : Tensor


  protected def doDeriveInputErrorForBatch(context: PredictContext,
                                           error:   Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //   State management.
  // ---------------------------------------------------------------------------
  final override def state
  : ZeroMeanState = ZeroMeanState(
    super.state,
    learningRate.state
  )

  final override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: ZeroMeanState =>
        learningRate.restoreState(state.learningRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class ZeroMeanBuilder
  extends MapLayerBuilder[ZeroMeanBuilder] {

  override def repr
  : ZeroMeanBuilder = this

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
  : ZeroMeanBuilder = {
    domain_=(value)
    this
  }

  private var _learningRate
  : ParameterBuilder = par.CMAFactorBuilder().clip(
    0.01f, Real.positiveInfinity
  )

  def learningRate: ParameterBuilder = _learningRate

  def learningRate_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _learningRate = value
  }

  def setLearningRate(value: ParameterBuilder)
  : ZeroMeanBuilder = {
    learningRate_=(value)
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
  : ZeroMeanBuilder = {
    runningMeanReference_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _domain :: _learningRate :: _runningMeanReference :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _domain.hashCode())
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _runningMeanReference.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ZeroMeanBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ZeroMeanBuilder =>
      _domain                == other._domain        &&
      _learningRate         == other._learningRate &&
      _runningMeanReference == other._runningMeanReference
    case _ =>
      false
  })

  override protected def doCopy()
  : ZeroMeanBuilder = ZeroMeanBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ZeroMeanBuilder =>
        other._domain               = _domain
        other._learningRate         = _learningRate
        other._runningMeanReference = _runningMeanReference
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
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

  override protected def doWeightLayoutFor(hints:   BuildHints,
                                           builder: TensorLayoutBufferBuilder)
  : Unit = {
    if (_runningMeanReference.segmentNo == 0 || !builder.contains(_runningMeanReference)) {
      val layout = runningMeanLayoutFor(hints.layout)
      builder.register(_runningMeanReference, layout)
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = ZeroMeanBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = ZeroMeanBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {
    runningMeanReference_=(fn(_runningMeanReference))
  }

}

object ZeroMeanBuilder
  extends ModuleVariantTable[ZeroMeanBuilder] {

  register(2, ZeroMean_JVM_Baseline_Description)

  final def apply()
  : ZeroMeanBuilder = new ZeroMeanBuilder

  final def apply(domain: TensorDomain)
  : ZeroMeanBuilder = apply().setDomain(domain)

  final def apply(domain:       TensorDomain,
                  learningRate: ParameterBuilder)
  : ZeroMeanBuilder = apply(
    domain
  ).setLearningRate(learningRate)

}

final case class ZeroMeanState(override val parent: InstanceState,
                               learningRate:        InstanceState)
  extends ModuleState {
}
