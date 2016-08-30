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

trait DependentTensor
  extends Tensor {

  override def createSibling()
  : DependentTensor

  override def createSibling(newLayout: TensorLayout)
  : DependentTensor

  override def createSiblingAndClear()
  : DependentTensor

  override def createSiblingAndClear(newLayout: TensorLayout)
  : DependentTensor

  override def copy
  : DependentTensor


  // ---------------------------------------------------------------------------
  //    Data access related.
  // ---------------------------------------------------------------------------
  override def platform
  : DependentPlatform

  override def layout
  : DependentTensorLayout


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(size: Size)
  : DependentTensor

  override def apply(index: Int)
  : DependentTensor

  override def apply(indices: Range)
  : DependentTensor

  override def concat(other: Tensor)
  : DependentTensor

  override def concat(other0: Tensor, others: Tensor*)
  : DependentTensor

  override def concat[T <: Tensor](others: Array[T])
  : DependentTensor

  override def +(value: Real)
  : DependentTensor

  override def +(other: Tensor)
  : DependentTensor

  override def unary_-()
  : DependentTensor

  override def -(other: Tensor)
  : DependentTensor

  override def *(value: Real)
  : DependentTensor

  override def :*(other: Tensor)
  : DependentTensor

  override def :/(other: Tensor)
  : DependentTensor

  override def reciprocal()
  : DependentTensor

  override def ++(other: Tensor)
  : DependentTensor

  override def :++(other: Tensor)
  : DependentTensor

  override def slice(tuple0: Int, noTuples: Int)
  : DependentTensor

  override def slice(tuple0: Int, size: Size)
  : DependentTensor

  override def sliceChannels(channel0: Int, noChannels: Int)
  : DependentTensor

  override def sliceChannels(channel0: Int, size: Size)
  : DependentTensor

}
