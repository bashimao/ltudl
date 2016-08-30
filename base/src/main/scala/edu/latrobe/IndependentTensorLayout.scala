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

import edu.latrobe.sizes._
import org.json4s.JsonAST._
import scala.util.hashing._

final class IndependentTensorLayout(override val size:      Size,
                                    override val noSamples: Int)
  extends TensorLayout {
  require(size != null && noSamples >= 0)

  override def toString
  : String = s"$size x $noSamples"

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, size.hashCode())
    tmp = MurmurHash3.mix(tmp, noSamples.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IndependentTensorLayout]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: IndependentTensorLayout =>
      size      == other.size &&
        noSamples == other.noSamples
    case _ =>
      false
  })

  override def noValues
  : Int = size.noValues * noSamples

  override def noTuples
  : Int = size.noTuples * noSamples

  override def concat(other: TensorLayout)
  : IndependentTensorLayout = {
    require(size == other.size)
    IndependentTensorLayout(
      size,
      noSamples + other.noSamples
    )
  }

  override def concat(others: Array[TensorLayout])
  : IndependentTensorLayout = ArrayEx.foldLeft(
    this,
    others
  )(_.concat(_))

  override def concat(others: Traversable[TensorLayout])
  : IndependentTensorLayout = others.foldLeft(
    this
  )(_.concat(_))

  override def offsetFor(sampleNo: Int)
  : Int = {
    require(
      sampleNo >= 0 &&
        sampleNo <  noSamples
    )
    sampleNo * size.noValues
  }

  override def offsetFor(sampleNo: Int, valueNo: Int)
  : Int = {
    val sampleSize = size.noValues
    require(
      sampleNo >= 0         &&
        sampleNo <  noSamples &&
        valueNo  >= 0         &&
        valueNo  <  sampleSize
    )
    valueNo + sampleNo * sampleSize
  }

  override def sampleFor(offset: Int)
  : Int = {
    val sampleSize = size.noValues
    require(
      offset >= 0 &&
        offset <  sampleSize * noSamples
    )
    offset / sampleSize
  }

  override def ++(other: TensorLayout)
  : IndependentTensorLayout = {
    require(noSamples == other.noSamples)
    IndependentTensorLayout(
      size ++ other.size,
      noSamples
    )
  }

  override def :++(other: TensorLayout)
  : IndependentTensorLayout = {
    require(noSamples == other.noSamples)
    IndependentTensorLayout(
      size :++ other.size,
      noSamples
    )
  }

  override def makeIndependent
  : IndependentTensorLayout = this

  override protected def doToJson()
  : List[JField] = List(
    Json.field("size",      size),
    Json.field("noSamples", noSamples)
  )


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override def toEdgeLabel
  : String = toString

}

object IndependentTensorLayout {

  final def apply(size: Size)
  : IndependentTensorLayout = apply(size, 1)

  final def apply(size: Size, noSamples: Int)
  : IndependentTensorLayout = new IndependentTensorLayout(size, noSamples)

  final def derive(noChannels: Int)
  : IndependentTensorLayout = derive(noChannels, 1)

  final def derive(noChannels: Int, noSamples: Int)
  : IndependentTensorLayout = apply(Size1(1, noChannels), noSamples)

  final def derive(json: JValue)
  : IndependentTensorLayout = derive(json.asInstanceOf[JObject])

  final def derive(json: JObject)
  : IndependentTensorLayout = {
    val fields = json.obj.toMap
    apply(
      Size.derive(fields("size")),
      Json.toInt(fields("noSamples"))
    )
  }

  final val one: IndependentTensorLayout = apply(Size1.one, 1)

  final val zero: IndependentTensorLayout = apply(Size1.zero, 0)

}
