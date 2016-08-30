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
import scala.collection._
import scala.util.hashing._

final class TensorLayoutBank(override val segments: SortedMap[Int, IndependentTensorLayout])
  extends BankEx[TensorLayoutBank, IndependentTensorLayout] {
  require(!segments.exists(_._2 == null))

  override def toString
  : String = s"TensorLayoutBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TensorLayoutBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TensorLayoutBank =>
      segments == other.segments
    case _ =>
      false
  })

  def noValues
  : Long = foldLeftSegments(
    0L
  )(_ + _.noValues)

  /*
  def apply(segmentNo: Int)
  : IndependentTensorLayout = segments.getOrElse(
    segmentNo,
    IndependentTensorLayout.zero
  )
  */


  // ---------------------------------------------------------------------------
  //   Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, IndependentTensorLayout])
  : TensorLayoutBank = TensorLayoutBank(banks)

  override protected def doToJson(segment: IndependentTensorLayout)
  : JValue = segment.toJson

  def toRealTensorBank(fillFn: IndependentTensorLayout => RealTensor)
  : ValueTensorBank = {
    val result = mapSegments(fillFn)
    ValueTensorBank(result)
  }

  /*
  def +(other: ParameterGroupLayout): ParameterGroupLayout = {
    val builder = ParameterGroupLayoutBuilder()
    val zipped = MapEx.zipValuesEx(segments, other.segments)(
      // If this throws an exception, you have no choice but to insert a section
      // separator because your model has more than 2B parameters.
      (a, b) => a ?+ b,
      a => a,
      b => b
    )
    ParameterGroupLayout(zipped)
  }

  def +(other: (Int, Int)): ParameterGroupLayout = {
    val kv = segments.get(other._1)
    if (kv.isDefined) {
      val newSegment = (other._1, kv.get ?+ other._2)
      ParameterGroupLayout(segments + newSegment)
    }
    else {
      ParameterGroupLayout(segments + other)
    }
  }
  */

}

object TensorLayoutBank
  extends BankExCompanion[TensorLayoutBank, IndependentTensorLayout] {

  final override def apply(segments: SortedMap[Int, IndependentTensorLayout])
  : TensorLayoutBank = new TensorLayoutBank(segments)

  final override protected def doDerive(json: JValue)
  : IndependentTensorLayout = IndependentTensorLayout.derive(json)

}

final class TensorLayoutBankBuilder
  extends BankExBuilder[TensorLayoutBank, IndependentTensorLayout] {

  override def result()
  : TensorLayoutBank = {
    val builder = SortedMap.newBuilder[Int, IndependentTensorLayout]
    builder ++= segments
    TensorLayoutBank(builder.result())
  }

}

object TensorLayoutBankBuilder {

  final def apply()
  : TensorLayoutBankBuilder = new TensorLayoutBankBuilder

}
