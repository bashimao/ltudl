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
import scala.util.hashing._

/**
  * A layer that multiplies each channel with a weight to scale it.
  *
  * f(x_i) = x_i * w_i
  *
  *       -1   x_i
  * f(x_i)   = ---
  *            w_i
  *
  * d f(x_a)
  * -------- = w_a
  *  d x_a
  *
  *   d f(x_a)
  * ----------- = 0
  * d x_b, a!=b
  *
  *            ---
  * D f(x_a)   \   d f(x_a)
  * -------- = /   -------- di = w_a da
  *  D x_a     ---   w_i
  *             i
  *
  * d f(x_a)
  * -------- = x_a
  *  d w_a
  *
  *   d f(x_a)
  * ----------- = 0
  * d w_b, a!=b
  *
  *            ---
  * D f(x_a)   \   d f(x_a)
  * -------- = /   -------- di = x_a da
  *  D w_a     ---   w_i
  *             i
  *
  */
abstract class MultiplyFilter
  extends MapLayerEx[MultiplyFilterBuilder]
    with FilterLayerLike[MultiplyFilterBuilder] {

  // ---------------------------------------------------------------------------
  //   Statistics
  // ---------------------------------------------------------------------------
  @transient
  final override lazy val noNeurons
  : Long = filterLayout.noValues


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
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
    filterReference.foreach(
      initializer(this, _, filter, inputFanSize, outputFanSize)
    )
  }

  final override def extractWeightsFor(neuronNo: Int)
  : Array[Real] = Array(filter.get(neuronNo))


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerValue(mode:           Mode,
                                                 inPlaceAllowed: Boolean,
                                                 input:          Tensor,
                                                 reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerValue(input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerUnit(mode:           Mode,
                                                inPlaceAllowed: Boolean,
                                                input:          Tensor,
                                                reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerUnit(input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerChannel(mode:           Mode,
                                                   inPlaceAllowed: Boolean,
                                                   input:          Tensor,
                                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerChannel(input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerSample(mode:           Mode,
                                                  inPlaceAllowed: Boolean,
                                                  input:          Tensor,
                                                  reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerSample(input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerBatch(mode:           Mode,
                                                 inPlaceAllowed: Boolean,
                                                 input:          Tensor,
                                                 reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerBatch(input)
    (out, EmptyContext)
  }

  protected def doPredictPerValue(input: Tensor)
  : Tensor

  protected def doPredictPerUnit(input: Tensor)
  : Tensor

  protected def doPredictPerChannel(input: Tensor)
  : Tensor

  protected def doPredictPerSample(input: Tensor)
  : Tensor

  protected def doPredictPerBatch(input: Tensor)
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
  final override protected def doDeriveFilterGradients(input:   Tensor,
                                                       context: PredictContext,
                                                       error:   Tensor,
                                                       sink:    ValueTensor)
  : Unit = domain match {
    case TensorDomain.Value =>
      require(error.layout == inputLayoutHint)
      doDeriveFilterGradientsPerValue(input, error, sink)

    case TensorDomain.Unit =>
      require(error.layout.size == inputSizeHint)
      doDeriveFilterGradientsPerUnit(input, error, sink)

    case TensorDomain.Channel =>
      require(error.layout.size.noChannels == inputSizeHint.noChannels)
      doDeriveFilterGradientsPerChannel(input, error, sink)

    case TensorDomain.Sample =>
      require(error.layout.noSamples == inputLayoutHint.noSamples)
      doDeriveFilterGradientsPerSample(input, error, sink)

    case TensorDomain.Batch =>
      doDeriveFilterGradientsPerBatch(input, error, sink)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doDeriveFilterGradientsPerValue(input: Tensor,
                                                error: Tensor,
                                                sink:  ValueTensor)
  : Unit

  protected def doDeriveFilterGradientsPerUnit(input: Tensor,
                                               error: Tensor,
                                               sink:  ValueTensor)
  : Unit

  protected def doDeriveFilterGradientsPerChannel(input: Tensor,
                                                  error: Tensor,
                                                  sink:  ValueTensor)
  : Unit

  protected def doDeriveFilterGradientsPerSample(input: Tensor,
                                                 error: Tensor,
                                                 sink:  ValueTensor)
  : Unit

  protected def doDeriveFilterGradientsPerBatch(input: Tensor,
                                                error: Tensor,
                                                sink:  ValueTensor)
  : Unit

  final override protected def doDeriveInputError(inputLayout: TensorLayout,
                                                  context:     PredictContext,
                                                  error:       Tensor)
  : Tensor = domain match {
    case TensorDomain.Value =>
      require(error.layout == inputLayoutHint)
      doDeriveInputErrorPerValue(error)

    case TensorDomain.Unit =>
      require(error.layout.size == inputSizeHint)
      doDeriveInputErrorPerUnit(error)

    case TensorDomain.Channel =>
      require(error.layout.size.noChannels == inputSizeHint.noChannels)
      doDeriveInputErrorPerChannel(error)

    case TensorDomain.Sample =>
      require(error.layout.noSamples == inputLayoutHint.noSamples)
      doDeriveInputErrorPerSample(error)

    case TensorDomain.Batch =>
      doDeriveInputErrorPerBatch(error)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doDeriveInputErrorPerValue(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerUnit(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerChannel(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerSample(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerBatch(error: Tensor)
  : Tensor

}

final class MultiplyFilterBuilder
  extends MapLayerExBuilder[MultiplyFilterBuilder]
    with FilterLayerLikeBuilder[MultiplyFilterBuilder] {

  override def repr
  : MultiplyFilterBuilder = this

  override def defaultDomain()
  : TensorDomain = TensorDomain.Channel

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MultiplyFilterBuilder]

  override protected def doCopy()
  : MultiplyFilterBuilder = MultiplyFilterBuilder()


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  override def filterLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = domain match {
    case TensorDomain.Value =>
      layoutHint.makeIndependent
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


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override protected def doWeightLayoutFor(hints:   BuildHints,
                                           builder: TensorLayoutBufferBuilder)
  : Unit = {
    if (filterReference.segmentNo == 0 || !builder.contains(filterReference)) {
      val layout = filterLayoutFor(hints.layout)
      builder.register(filterReference, layout)
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = MultiplyFilterBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = MultiplyFilterBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

}

object MultiplyFilterBuilder
  extends ModuleVariantTable[MultiplyFilterBuilder] {

  register(2, ImmediateFilter_JVM_Baseline_Description)

  final def apply()
  : MultiplyFilterBuilder = new MultiplyFilterBuilder

  final def apply(domain: TensorDomain)
  : MultiplyFilterBuilder = apply().setDomain(domain)

}
