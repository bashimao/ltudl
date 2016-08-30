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
import edu.latrobe.sizes._
import scala.util.hashing._

abstract class LinearFilterLike[TBuilder <: LinearFilterLikeBuilder[_]]
  extends Layer[TBuilder]
    with FilterLayerLike[TBuilder] {

  /**
    * Must override with lazy val.
    */
  def outputPlatform
  : IndependentPlatform

  final val outputSize
  : Size1 = builder.outputSize

  final val outputLayout
  : IndependentTensorLayout = inputLayoutHint.derive(outputSize)

  final override val outputHints
  : BuildHints = inputHints.derive(outputPlatform, outputLayout)


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons
  : Long = outputSize.noValues


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    require(input.layout.size == inputSizeHint)
    val out = doPredict(input)
    (out, EmptyContext)
  }

  protected def doPredict(input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveFilterGradients(input:   Tensor,
                                                       context: PredictContext,
                                                       error:   Tensor,
                                                       sink:    ValueTensor)
  : Unit = {
    require(error.layout.size == outputSize)
    doDeriveFilterGradients(input, error, sink)
  }

  protected def doDeriveFilterGradients(input: Tensor,
                                        error: Tensor,
                                        sink:  Tensor)
  : Unit


  final override protected def doDeriveInputError(inputLayout: TensorLayout,
                                                  context:     PredictContext,
                                                  error:       Tensor)
  : Tensor = {
    require(error.layout.size == outputSize)
    doDeriveInputError(error)
  }

  protected def doDeriveInputError(error: Tensor)
  : Tensor

}

abstract class LinearFilterLikeBuilder[TThis <: LinearFilterLikeBuilder[_]]
  extends LayerBuilder[TThis]
    with FilterLayerLikeBuilder[TThis] {

  final private var _noOutputs
  : Int = 1

  final def noOutputs
  : Int = _noOutputs

  final def noOutputs_=(value: Int)
  : Unit = {
    require(value > 0)
    _noOutputs = value
  }

  final def setNoOutputs(value: Int)
  : TThis = {
    noOutputs_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _noOutputs :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _noOutputs.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LinearFilterBuilder =>
      _noOutputs == other._noOutputs
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: LinearFilterBuilder =>
        other._noOutputs = _noOutputs
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  def filterSizeFor(sizeHint: Size)
  : Size = {
    require(sizeHint.noTuples == 1)
    Size1(1, sizeHint.noValues)
  }

  final override def filterLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout(
    filterSizeFor(layoutHint.size),
    _noOutputs
  )


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  final override def weightLayoutFor(hints:   BuildHints,
                                     builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    if (filterReference.segmentNo == 0 || !builder.contains(filterReference)) {
      val layout = filterLayoutFor(hints.layout)
      builder.register(filterReference, layout)
    }
    outputHintsFor(hints)
  }

  def outputPlatformFor(hints: BuildHints)
  : Platform

  def outputSize
  : Size1 = Size1(1, _noOutputs)

  final def outputSizeFor(sizeHint: Size)
  : Size1 = sizeHint match {
    case size: Size1 =>
      outputSize
    case _ =>
      throw new MatchError(sizeHint)
  }

  final def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSizeFor(layoutHint.size))

  final override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val platform = outputPlatformFor(hints)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

}
