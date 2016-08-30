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
import scala.collection.Map
import scala.util.hashing._

trait RealTensor
  extends Tensor
    with ValueTensor {

  final override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, layout.hashCode())
    tmp = MurmurHash3.mix(tmp, ArrayEx.hashCode(values))
    tmp
  }

  final override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Tensor =>
      layout == other.layout &&
        ArrayEx.compare(values, other.values)
    case _ =>
      false
  })

  override def createSibling()
  : RealTensor

  override def createSibling(newLayout: TensorLayout)
  : RealTensor

  override def createSiblingAndClear()
  : RealTensor

  override def createSiblingAndClear(newLayout: TensorLayout)
  : RealTensor

  override def copy
  : RealTensor


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(size: Size)
  : RealTensor

  override def apply(index: Int)
  : RealTensor

  override def apply(indices: Range)
  : RealTensor

  override def concat(other: Tensor)
  : RealTensor

  override def concat(other0: Tensor, others: Tensor*)
  : RealTensor

  override def concat[T <: Tensor](others: Array[T])
  : RealTensor

  override def +(value: Real)
  : RealTensor

  override def +(other: Tensor)
  : RealTensor

  override def unary_-()
  : RealTensor

  override def -(other: Tensor)
  : RealTensor

  override def *(value: Real)
  : RealTensor

  override def :*(other: Tensor)
  : RealTensor

  override def :/(other: Tensor)
  : RealTensor

  /**
    * Compute the reciprocal.
    */
  override def reciprocal()
  : RealTensor

  override def ++(other: Tensor)
  : RealTensor

  override def :++(other: Tensor)
  : RealTensor

  override def slice(tuple0: Int, noTuples: Int)
  : RealTensor

  override def slice(tuple0: Int, size: Size)
  : RealTensor

  override def sliceChannels(channel0: Int, noChannels: Int)
  : RealTensor

  override def sliceChannels(channel0: Int, size: Size)
  : RealTensor


}

object RealTensor
  extends JsonSerializableCompanionEx[RealArrayTensor] {

  final override def derive(fields: Map[String, JValue])
  : RealArrayTensor = RealArrayTensor(
    IndependentTensorLayout.derive(fields("layout")),
    Json.toRealArray(fields("values"))
  )

}
