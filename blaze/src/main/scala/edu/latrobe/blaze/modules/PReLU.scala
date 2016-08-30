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

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._

// TODO: This code has not been tested yet!
abstract class PReLU
  extends MapLayer[PReLUBuilder]
    with TrainableLayer[PReLUBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons
  : Long = inputSizeHint.noValues


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  final val pReLULayout
  : IndependentTensorLayout = builder.pReLULayoutFor(inputSizeHint)

  def pReLUReference
  : Option[LabeledBufferReference]

  def pReLU
  : ValueTensor

  @transient
  final override lazy val weightReferences
  : Set[LabeledBufferReference] = {
    val builder = Set.newBuilder[LabeledBufferReference]
    pReLUReference.map(builder += _)
    builder.result()
  }

  final override def extractWeightsFor(neuronNo: Long)
  : Array[Real] = {
    if (neuronNo >= 0L && neuronNo <= Int.MaxValue) {
      Array(pReLU.get(neuronNo.toInt))
    }
    else {
      throw new IndexOutOfBoundsException
    }
  }

  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize  = 1
    val outputFanSize = 1
    pReLUReference.foreach(
      initializer(this, _, pReLU, inputFanSize, outputFanSize)
    )
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    require(input.layout.size == pReLU.layout.size)
    val out = doPredict(input)
    (out, EmptyContext)
  }

  protected def doPredict(input: Tensor): Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


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
    require(error.layout.size == pReLU.layout.size)

    // Compute gradients depending on group selection.
    pReLUReference.foreach(pr => {
      val s = sink.get(pr)
      s.foreach(doDerivePReLUGradients(input, error, _))
    })
  }

  protected def doDerivePReLUGradients(input: Tensor,
                                       error: Tensor,
                                       sink:  ValueTensor)
  : Unit

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = {
    require(error.layout.size == pReLU.layout.size)
    doDeriveInputError(input, error)
  }

  protected def doDeriveInputError(input: Tensor,
                                   error: Tensor)
  : Tensor

}

final class PReLUBuilder
  extends MapLayerBuilder[PReLUBuilder]
    with TrainableLayerBuilder[PReLUBuilder] {

  override def repr
  : PReLUBuilder = this

  /**
    * bank    >= 0
    * segment =  0 -> Automatically assign a segment number.
    *         >  1 -> Use fixed segment number. (use this to link weights)
    */
  private var _pReLUReference
  : LabeledBufferReference = LabeledBufferReference("pReLU")

  def pReLUReference
  : LabeledBufferReference = _pReLUReference

  def pReLUReference_=(value: LabeledBufferReference)
  : Unit = {
    require(value != null)
    _pReLUReference = value
  }


  def setPReLUReference(value: LabeledBufferReference)
  : PReLUBuilder = {
    pReLUReference_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _pReLUReference :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PReLUBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PReLUBuilder =>
      _pReLUReference == other._pReLUReference
    case _ =>
      false
  })

  override protected def doCopy()
  : PReLUBuilder = PReLUBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PReLUBuilder =>
        other._pReLUReference = _pReLUReference
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  def pReLULayoutFor(sizeHint: Size)
  : IndependentTensorLayout = IndependentTensorLayout(sizeHint, 1)

  override protected def doWeightLayoutFor(hints:   BuildHints,
                                           builder: TensorLayoutBufferBuilder)
  : Unit = {
    if (_pReLUReference.segmentNo == 0 || !builder.contains(_pReLUReference)) {
      val layout = pReLULayoutFor(hints.layout.size)
      builder.register(_pReLUReference, layout)
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = PReLUBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = PReLUBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = pReLUReference_=(fn(_pReLUReference))

}

object PReLUBuilder
  extends ModuleVariantTable[PReLUBuilder] {

  register(2, PReLU_JVM_Baseline_Description)

  final def apply(): PReLUBuilder = new PReLUBuilder

}
