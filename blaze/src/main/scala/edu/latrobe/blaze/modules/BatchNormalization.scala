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

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.generic._
import edu.latrobe.blaze.{parameters => par}
import java.util.UUID
import scala.collection._
import scala.util.hashing._

abstract class BatchNormalization
  extends MapLayer[BatchNormalizationBuilder]
    with TrainableLayer[BatchNormalizationBuilder]
    with NonPenalizing {

  final val learningRate
  : Parameter = builder.learningRate.build("BNLR", seed)

  final val epsilon
  : Real = builder.epsilon

  final val noChannels
  : Int = inputSizeHint.noChannels

  final val runningMeanLayout
  : IndependentTensorLayout = builder.runningMeanLayoutFor(inputLayoutHint)

  def runningMeanReference
  : Option[LabeledBufferReference]

  def runningMean
  : ValueTensor

  final val runningVarianceLayout
  : IndependentTensorLayout = builder.runningVarianceLayoutFor(inputLayoutHint)

  def runningVarianceReference: Option[LabeledBufferReference]

  def runningVariance
  : ValueTensor

  final val filterLayout
  : IndependentTensorLayout = builder.filterLayoutFor(inputLayoutHint)

  def filterReference
  : Option[LabeledBufferReference]

  def filter
  : ValueTensor

  final val biasLayout
  : IndependentTensorLayout = builder.biasLayoutFor(inputLayoutHint)

  def biasReference
  : Option[LabeledBufferReference]

  def bias
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

  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize  = inputSizeHint.noTuples
    val outputFanSize = outputHints.layout.noTuples

    runningMeanReference.foreach(
      initializer(this, _, runningMean, inputFanSize, outputFanSize)
    )
    runningVarianceReference.foreach(
      initializer(this, _, runningVariance, inputFanSize, outputFanSize)
    )
    filterReference.foreach(
      initializer(this, _, filter, inputFanSize, outputFanSize)
    )
    biasReference.foreach(
      initializer(this, _, bias, inputFanSize, outputFanSize)
    )
  }


  // ---------------------------------------------------------------------------
  //    Statistics
  // ---------------------------------------------------------------------------
  final override val noNeurons
  : Long = {
    val rm  = runningMeanLayout.noValues
    val rvl = runningVarianceLayout.noValues
    val fr  = filterLayout.noValues
    val br  = biasLayout.noValues
    rm + rvl + fr + br
  }


  // ---------------------------------------------------------------------------
  //    Weights related
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
    n -= runningVarianceLayout.noValues
    if (n < filterLayout.noValues) {
      return Array(filter.get(n.toInt))
    }
    n -= filterLayout.noValues
    if (n < biasLayout.noValues) {
      return Array(bias.get(n.toInt))
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
  : (Tensor, PredictContext) = {
    require(input.layout.size.noChannels == noChannels)

    mode match {
      case mode: Training =>
        // Compute learning rate.
        val lr = learningRate.get(mode.phaseNo, RealRange.zeroToInfinity)
        val result = doPredictForTraining(input, lr)
        //learningRate.update(mode.phaseNo, Real.nan)
        result

      case mode: Inference =>
        val out = doPredictForInference(input)
        (out, EmptyContext)

      case _ =>
        throw new MatchError(mode)
    }
  }

  protected def doPredictForTraining(input:        Tensor,
                                     learningRate: Real)
  : (Tensor, PredictContext)

  protected def doPredictForInference(input: Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveWeightGradients(input:     Tensor,
                                                       reference: Tensor,
                                                       output:    Tensor,
                                                       context:   PredictContext,
                                                       error:     Tensor,
                                                       sink:      ValueTensorBuffer)
  : Unit = {
    require(error.layout.size.noChannels == noChannels)

    val fr         = filterReference.orNull
    val filterSink = if (fr != null) sink.get(fr) else None
    val br         = biasReference.orNull
    val biasSink   = if (br != null) sink.get(br) else None

    doDeriveWeightGradients(
      input,
      context,
      error,
      filterSink,
      biasSink
    )
  }

  protected def doDeriveWeightGradients(input:      Tensor,
                                        context:    PredictContext,
                                        error:      Tensor,
                                        filterSink: Option[ValueTensor],
                                        biasSink:   Option[ValueTensor])
  : Unit

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = {
    require(error.layout.size.noChannels == noChannels)
    doDeriveInputError(input, context, error)
  }

  protected def doDeriveInputError(input:   Tensor,
                                   context: PredictContext,
                                   error:   Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //   State management.
  // ---------------------------------------------------------------------------
  override def state
  : BatchNormalizationState = BatchNormalizationState(
    super.state,
    learningRate.state
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: BatchNormalizationState =>
        learningRate.restoreState(state.learningRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

/**
  * Implementation of batch normalization as described in:
  * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
  * (Ioffe & Szegedy, 2015)
  *
  * Check Normalization.scala for equations.
  *
  * Differences:
  * Like in the paper we sometimes use population figures instead of sample
  * figures. Furthermore, like described in the paper we keep
  *
  */
final class BatchNormalizationBuilder
  extends MapLayerBuilder[BatchNormalizationBuilder]
    with TrainableLayerBuilder[BatchNormalizationBuilder] {

  override def repr
  : BatchNormalizationBuilder = this

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
  : BatchNormalizationBuilder = {
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
  : BatchNormalizationBuilder = {
    epsilon_=(value)
    this
  }

  private var _runningMeanReference
  : LabeledBufferReference = LabeledBufferReference(1000, 0, "runningMean")

  def runningMeanReference
  : LabeledBufferReference = _runningMeanReference

  def runningMeanReference_=(value: LabeledBufferReference): Unit = {
    require(value != null)
    _runningMeanReference = value
  }

  def setRunningMeanReference(value: LabeledBufferReference)
  : BatchNormalizationBuilder = {
    runningMeanReference_=(value)
    this
  }

  private var _runningVarianceReference
  : LabeledBufferReference = LabeledBufferReference(1000, 0, "runningVariance")

  def runningVarianceReference
  : LabeledBufferReference = _runningVarianceReference

  def runningVarianceReference_=(value: LabeledBufferReference): Unit = {
    require(value != null)
    _runningVarianceReference = value
  }

  def setRunningVarianceReference(value: LabeledBufferReference)
  : BatchNormalizationBuilder = {
    runningVarianceReference_=(value)
    this
  }

  /**
    * bank    >= 0
    * segment =  0 -> Automatically assign a segment number.
    *         >  1 -> Use fixed segment number. (use this to link weights)
    */
  private var _filterReference
  : LabeledBufferReference = LabeledBufferReference("gamma")

  def filterReference
  : LabeledBufferReference = _filterReference

  def filterReference_=(value: LabeledBufferReference): Unit = {
    require(value != null)
    _filterReference = value
  }

  def setFilterReference(value: LabeledBufferReference)
  : BatchNormalizationBuilder = {
    filterReference_=(value)
    repr
  }

  /**
    * = 0 -> Automatically assign a segment number.
    * > 1 -> Use fixed segment number. (use this to link weights)
    **/
  private var _biasReference
  : LabeledBufferReference = LabeledBufferReference("beta")

  def biasReference
  : LabeledBufferReference = _biasReference

  def biasReference_=(value: LabeledBufferReference): Unit = {
    require(value != null)
    _biasReference = value
  }

  def setBiasReference(value: LabeledBufferReference)
  : BatchNormalizationBuilder = {
    biasReference_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _learningRate :: f"${_epsilon}%.4g" :: _runningMeanReference :: _runningVarianceReference :: _filterReference :: _biasReference :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp = MurmurHash3.mix(tmp, _runningMeanReference.hashCode())
    tmp = MurmurHash3.mix(tmp, _runningVarianceReference.hashCode())
    tmp = MurmurHash3.mix(tmp, _filterReference.hashCode())
    tmp = MurmurHash3.mix(tmp, _biasReference.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BatchNormalizationBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BatchNormalizationBuilder =>
      _learningRate             == other._learningRate             &&
      _epsilon                  == other._epsilon                  &&
      _runningMeanReference     == other._runningMeanReference     &&
      _runningVarianceReference == other._runningVarianceReference &&
      _filterReference          == other._filterReference          &&
      _biasReference            == other._biasReference
    case _ =>
      false
  })

  override protected def doCopy()
  : BatchNormalizationBuilder = BatchNormalizationBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: BatchNormalizationBuilder =>
        other._learningRate             = _learningRate
        other._epsilon                  = _epsilon
        other._runningMeanReference     = _runningMeanReference
        other._runningVarianceReference = _runningVarianceReference
        other._filterReference          = _filterReference
        other._biasReference            = _biasReference
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  def runningMeanLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout.derive(layoutHint.size.noChannels)

  def runningVarianceLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout.derive(layoutHint.size.noChannels)

  def filterLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout.derive(layoutHint.size.noChannels)

  def biasLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout.derive(layoutHint.size.noChannels)

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
    if (_filterReference.segmentNo == 0 || !builder.contains(_filterReference)) {
      val layout = filterLayoutFor(hints.layout)
      builder.register(_filterReference, layout)
    }
    if (_biasReference.segmentNo == 0 || !builder.contains(_biasReference)) {
      val layout = biasLayoutFor(hints.layout)
      builder.register(_biasReference, layout)
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = BatchNormalizationBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = BatchNormalizationBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {
    runningMeanReference_=(fn(_runningMeanReference))
    runningVarianceReference_=(fn(_runningVarianceReference))
    filterReference_=(fn(_filterReference))
    biasReference_=(fn(_biasReference))
  }

}

object BatchNormalizationBuilder
  extends ModuleVariantTable[BatchNormalizationBuilder] {

  register(64, BatchNormalization_Generic_Baseline_Description)

  final def apply()
  : BatchNormalizationBuilder = new BatchNormalizationBuilder

  final def apply(learningRate: ParameterBuilder)
  : BatchNormalizationBuilder = apply().setLearningRate(learningRate)

  final def apply(learningRate: ParameterBuilder,
                  epsilon:      Real)
  : BatchNormalizationBuilder = apply(
    learningRate
  ).setEpsilon(epsilon)

}

final case class BatchNormalizationState(override val parent: InstanceState,
                                         learningRate:        InstanceState)
  extends ModuleState {
}
