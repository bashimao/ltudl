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

package edu.latrobe

import org.json4s.JsonAST._

trait ValueTensor
  extends Tensor
    with IndependentTensor {

  override def createSibling()
  : ValueTensor

  override def createSibling(newLayout: TensorLayout)
  : ValueTensor

  override def createSiblingAndClear()
  : ValueTensor

  override def createSiblingAndClear(newLayout: TensorLayout)
  : ValueTensor

  override def copy
  : ValueTensor


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(size: Size)
  : ValueTensor

  override def apply(index: Int)
  : ValueTensor

  override def apply(indices: Range)
  : ValueTensor

  override def concat(other: Tensor)
  : ValueTensor

  override def concat(other0: Tensor, others: Tensor*)
  : ValueTensor

  override def concat[T <: Tensor](others: Array[T])
  : ValueTensor

  override def +(value: Real)
  : ValueTensor

  override def +(other: Tensor)
  : ValueTensor

  override def unary_-()
  : ValueTensor

  override def -(other: Tensor)
  : ValueTensor

  override def *(value: Real)
  : ValueTensor

  override def :*(other: Tensor)
  : ValueTensor

  override def :/(other: Tensor)
  : ValueTensor

  override def reciprocal()
  : ValueTensor


  // ---------------------------------------------------------------------------
  //    Fancy operations.
  // ---------------------------------------------------------------------------
  final override def approximateMean(rng:          PseudoRNG,
                                     noSamplesMax: Int)
  : Mean = Mean.derive(values, rng, noSamplesMax)

  final override def approximateMeanAndVariance(rng:          PseudoRNG,
                                                noSamplesMax: Int)
  : MeanAndVariance = MeanAndVariance.derive(values, rng, noSamplesMax)


  // ---------------------------------------------------------------------------
  //    Conversion & extraction methods.
  // ---------------------------------------------------------------------------
  final override protected def doToJson()
  : List[JField] = List(
    Json.field("layout", layout),
    Json.field("values", values)
  )

  final override def toValueTensor
  : ValueTensor = copy

  final override def asOrToValueTensor
  : ValueTensor = this

}
