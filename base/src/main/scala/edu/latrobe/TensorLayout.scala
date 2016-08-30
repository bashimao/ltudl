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

import scala.collection._

abstract class TensorLayout
  extends Equatable
    with Serializable
    with JsonSerializable {

  def size: Size

  def noSamples: Int

  def noValues: Int

  def noTuples: Int

  final def derive(size: Size)
  : IndependentTensorLayout = IndependentTensorLayout(size, noSamples)

  final def derive(noSamples: Int)
  : IndependentTensorLayout = IndependentTensorLayout(size, noSamples)

  def concat(other: TensorLayout): TensorLayout

  def concat(others: Array[TensorLayout]): TensorLayout

  def concat(others: Traversable[TensorLayout]): TensorLayout

  def makeIndependent: IndependentTensorLayout

  def offsetFor(sampleNo: Int): Int

  def offsetFor(sampleNo: Int, valueNo: Int): Int

  def sampleFor(offset: Int): Int

  def ++(other: TensorLayout): TensorLayout

  def :++(other: TensorLayout): TensorLayout


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  def toEdgeLabel
  : String

}
