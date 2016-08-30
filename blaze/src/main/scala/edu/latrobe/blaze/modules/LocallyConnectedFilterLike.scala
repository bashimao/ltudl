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
import scala.util.hashing._

abstract class LocallyConnectedFilterLike[TBuilder <: LocallyConnectedFilterLikeBuilder[_]]
  extends Layer[TBuilder]
    with FilterLayerLike[TBuilder] {

  final lazy val kernel: Kernel = builder.kernel

  final lazy val noMaps: Int = builder.noMaps

  /**
    * Must override with lazy val.
    */
  def outputPlatform
  : IndependentPlatform

  /**
    * Must override with lazy val.
    */
  def outputSize
  : Size

  final lazy val outputLayout
  : IndependentTensorLayout = inputLayoutHint.derive(outputSize)

  final override lazy val outputHints
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
                                        sink:  ValueTensor)
  : Unit

  override protected def doDeriveInputError(inputLayout: TensorLayout,
                                            context:     PredictContext,
                                            error:       Tensor)
  : Tensor = {
    require(error.layout.size == outputSize)
    doDeriveInputError(error)
  }

  protected def doDeriveInputError(error: Tensor)
  : Tensor

}

abstract class LocallyConnectedFilterLikeBuilder[TThis <: LocallyConnectedFilterLikeBuilder[_]]
  extends LayerBuilder[TThis]
    with FilterLayerLikeBuilder[TThis] {

  final private var _kernel
  : Kernel = _

  final def kernel
  : Kernel = _kernel

  final def kernel_=(value: Kernel)
  : Unit = {
    require(value != null)
    _kernel = value
  }

  final def setKernel(value: Kernel)
  : TThis = {
    kernel_=(value)
    repr
  }

  final private var _noMaps
  : Int = 1

  final def noMaps
  : Int = _noMaps

  final def noMaps_=(value: Int)
  : Unit = {
    require(value > 0)
    _noMaps = value
  }

  final def setNoMaps(value: Int)
  : TThis = {
    noMaps_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = s"${_kernel} x ${_noMaps}" :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _kernel.hashCode())
    tmp = MurmurHash3.mix(tmp, _noMaps.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LocallyConnectedFilterLikeBuilder[_] =>
      _kernel == other._kernel &&
      _noMaps == other._noMaps
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: LocallyConnectedFilterLikeBuilder[_] =>
        other._kernel = _kernel
        other._noMaps = _noMaps
      case _ =>
    }
  }


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

}