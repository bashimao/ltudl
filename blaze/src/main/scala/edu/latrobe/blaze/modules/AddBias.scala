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
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.sizes._
import scala.collection._
import scala.util.hashing._

/**
  * A simple layer that just adds a bias to the activations it receives.
  *
  * f(x_i) = x_i + w_i
  *
  *       -1
  * f(x_i)   = x_i - w_i
  *
  * d f(x_a)
  * -------- = 1
  *  d x_a
  *
  *   d f(x_a)
  * ----------- = 0
  * d x_b, a!=b
  *
  *            ---
  * D f(x_a)   \   d f(x_a)
  * -------- = /   -------- di = 1 da = da
  *  D x_a     ---   w_i
  *             i
  *
  * d f(x_a)
  * -------- = 1
  *  d x_a
  *
  *   d f(x_a)
  * ----------- = 0
  * d x_b, a!=b
  *
  *            ---
  * D f(x_a)   \   d f(x_a)
  * -------- = /   -------- di = 1 da = da
  *  D w_a     ---   w_i
  *             i
  *
  */
abstract class AddBias
  extends MapLayerEx[AddBiasBuilder]
    with TrainableLayer[AddBiasBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
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
    biasReference.map(builder += _)
    builder.result()
  }

  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize = domain match {
      case TensorDomain.Value   => 1
      case TensorDomain.Unit    => inputLayoutHint.noSamples
      case TensorDomain.Channel => inputLayoutHint.noTuples
      case TensorDomain.Sample  => inputSizeHint.noValues
      case TensorDomain.Batch   => inputLayoutHint.noValues
    }
    val outputFanSize = domain match {
      case TensorDomain.Value   => 1
      case TensorDomain.Unit    => outputHints.layout.noSamples
      case TensorDomain.Channel => outputHints.layout.noTuples
      case TensorDomain.Sample  => outputHints.layout.size.noValues
      case TensorDomain.Batch   => outputHints.layout.noValues
    }
    biasReference.foreach(
      initializer(this, _, bias, inputFanSize, outputFanSize)
    )
  }

  final override def extractWeightsFor(neuronNo: Long): Array[Real] = {
    if (neuronNo >= 0L && neuronNo <= Int.MaxValue) {
      Array(bias.get(neuronNo.toInt))
    }
    else {
      throw new IndexOutOfBoundsException
    }
  }


  // ---------------------------------------------------------------------------
  //   Statistics
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons
  : Long = biasLayout.noValues


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerValue(mode:           Mode,
                                                 inPlaceAllowed: Boolean,
                                                 input:          Tensor,
                                                 reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerValue(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerUnit(mode:           Mode,
                                                inPlaceAllowed: Boolean,
                                                input:          Tensor,
                                                reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerUnit(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerChannel(mode:           Mode,
                                                   inPlaceAllowed: Boolean,
                                                   input:          Tensor,
                                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerChannel(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerSample(mode:           Mode,
                                                  inPlaceAllowed: Boolean,
                                                  input:          Tensor,
                                                  reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerSample(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerBatch(mode:           Mode,
                                                 inPlaceAllowed: Boolean,
                                                 input:          Tensor,
                                                 reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerBatch(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  protected def doPredictPerValue(inPlaceAllowed: Boolean,
                                  input:          Tensor)
  : Tensor

  protected def doPredictPerUnit(inPlaceAllowed: Boolean,
                                 input:          Tensor)
  : Tensor

  protected def doPredictPerChannel(inPlaceAllowed: Boolean,
                                    input:          Tensor)
  : Tensor

  protected def doPredictPerSample(inPlaceAllowed: Boolean,
                                   input:          Tensor)
  : Tensor

  protected def doPredictPerBatch(inPlaceAllowed: Boolean,
                                  input:          Tensor)
  : Tensor

  final override protected def doPredictInvPerValue(output:  Tensor,
                                                    context: PredictContext)
  : Tensor = doPredictInvPerValue(output)

  final override protected def doPredictInvPerUnit(output:  Tensor,
                                                   context: PredictContext)
  : Tensor = doPredictInvPerUnit(output)

  final override protected def doPredictInvPerChannel(output:  Tensor,
                                                      context: PredictContext)
  : Tensor = doPredictInvPerChannel(output)

  final override protected def doPredictInvPerSample(output:  Tensor,
                                                     context: PredictContext)
  : Tensor = doPredictInvPerSample(output)

  final override protected def doPredictInvPerBatch(output:  Tensor,
                                                    context: PredictContext)
  : Tensor = doPredictInvPerBatch(output)

  protected def doPredictInvPerValue(output: Tensor)
  : Tensor

  protected def doPredictInvPerUnit(output: Tensor)
  : Tensor

  protected def doPredictInvPerChannel(output: Tensor)
  : Tensor

  protected def doPredictInvPerSample(output: Tensor)
  : Tensor

  protected def doPredictInvPerBatch(output: Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveWeightGradients(input:     Tensor,
                                                       reference: Tensor,
                                                       output:    Tensor,
                                                       context:   PredictContext,
                                                       error:     Tensor,
                                                       sink:      ValueTensorBuffer)
  : Unit = {
    // Compute gradients depending on group selection.
    biasReference.foreach(br => {
      val s = sink.get(br)
      s.foreach(doDeriveWeightGradients(error, _))
    })
  }

  final protected def doDeriveWeightGradients(error: Tensor,
                                              sink:  ValueTensor)
  : Unit = domain match {
    case TensorDomain.Value =>
      require(error.layout == inputLayoutHint)
      doDeriveWeightGradientsPerValue(error, sink)

    case TensorDomain.Unit =>
      require(error.layout.size == inputSizeHint)
      doDeriveWeightGradientsPerUnit(error, sink)

    case TensorDomain.Channel =>
      require(error.layout.size.noChannels == inputSizeHint.noChannels)
      doDeriveWeightGradientsPerChannel(error, sink)

    case TensorDomain.Sample =>
      require(error.layout.noSamples == inputLayoutHint.noSamples)
      doDeriveWeightGradientsPerSample(error, sink)

    case TensorDomain.Batch =>
      doDeriveWeightGradientsPerBatch(error, sink)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doDeriveWeightGradientsPerValue(error: Tensor,
                                                sink:  ValueTensor)
  : Unit

  protected def doDeriveWeightGradientsPerUnit(error: Tensor,
                                               sink:  ValueTensor)
  : Unit

  protected def doDeriveWeightGradientsPerChannel(error: Tensor,
                                                  sink:  ValueTensor)
  : Unit

  protected def doDeriveWeightGradientsPerSample(error: Tensor,
                                                 sink:  ValueTensor)
  : Unit

  protected def doDeriveWeightGradientsPerBatch(error: Tensor,
                                                sink:  ValueTensor)
  : Unit

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = error

}

final class AddBiasBuilder
  extends MapLayerExBuilder[AddBiasBuilder]
    with TrainableLayerBuilder[AddBiasBuilder] {

  override def repr
  : AddBiasBuilder = this

  override def defaultDomain()
  : TensorDomain = TensorDomain.Channel

  private var _biasReference
  : LabeledBufferReference = LabeledBufferReference("bias")

  def biasReference
  : LabeledBufferReference = _biasReference

  def biasReference_=(value: LabeledBufferReference): Unit = {
    require(value != null)
    _biasReference = value
  }

  def setBiasReference(value: LabeledBufferReference)
  : AddBiasBuilder = {
    biasReference_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _biasReference :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _biasReference.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AddBiasBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AddBiasBuilder =>
      _biasReference == other._biasReference
    case _ =>
      false
  })

  override protected def doCopy()
  : AddBiasBuilder = AddBiasBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AddBiasBuilder =>
        other._biasReference = _biasReference
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  def biasLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint match {
    case layoutHint: IndependentTensorLayout =>
      domain match {
        case TensorDomain.Value =>
          layoutHint
        case TensorDomain.Unit =>
          layoutHint.derive(1)
        case TensorDomain.Channel =>
          IndependentTensorLayout.derive(layoutHint.size.noChannels)
        case TensorDomain.Sample =>
          layoutHint.derive(Size1.one)
        case TensorDomain.Batch =>
          IndependentTensorLayout.one
        case _ =>
          throw new MatchError(domain)
      }
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doWeightLayoutFor(hints:   BuildHints,
                                           builder: TensorLayoutBufferBuilder)
  : Unit = {
    if (_biasReference.segmentNo == 0 || !builder.contains(_biasReference)) {
      val layout = biasLayoutFor(hints.layout)
      builder.register(_biasReference, layout)
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = AddBiasBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = AddBiasBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {
    biasReference_=(fn(_biasReference))
  }

}

object AddBiasBuilder
  extends ModuleVariantTable[AddBiasBuilder] {

  register(2, AddBias_JVM_Baseline_Description)

  final def apply()
  : AddBiasBuilder = new AddBiasBuilder

  final def apply(domain: TensorDomain)
  : AddBiasBuilder = apply().setDomain(domain)

}
