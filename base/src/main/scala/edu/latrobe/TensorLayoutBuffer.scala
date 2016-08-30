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

final class TensorLayoutBuffer(override val banks: SortedMap[Int, TensorLayoutBank])
  extends BufferEx[TensorLayoutBuffer, TensorLayoutBank, IndependentTensorLayout] {
  require(!banks.exists(_._2 == null))

  override def toString
  : String = s"LayoutBuffer[${banks.size}]"

  override def hashCode()
  : Int =  MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TensorLayoutBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TensorLayoutBuffer =>
      banks == other.banks
    case _ =>
      false
  })

  def noValues
  : Long = foldLeftBanks(
    0L
  )(_ + _.noValues)

  /*
  def apply(bankNo: Int)
  : TensorLayoutBank = banks.getOrElse(
    bankNo,
    TensorLayoutBank.empty
  )
  */


  // ---------------------------------------------------------------------------
  //    Operations
  // ---------------------------------------------------------------------------

  /*
  def foreachSegment(fn: (Int, Int, Int) => Unit)
  : Unit = foreachGroup(
    (i, g) => g.foreachSegment(fn(i, _, _))
  )
  */

  /*
  def +(other: ParameterBufferLayout)
  : ParameterBufferLayout = ParameterBufferLayout(
    MapEx.zipValuesEx(groups, other.groups)(
      (a, b) => a + b,
      a => a,
      b => b
    )
  )

  def +(other: (Int, ParameterGroupLayout)): ParameterBufferLayout = {
    val kv = groups.get(other._1)
    if (kv.isDefined) {
      val newGroup = (other._1, kv.get + other._2)
      ParameterBufferLayout(groups + newGroup)
    }
    else {
      ParameterBufferLayout(groups + other)
    }
  }
  */

  // ---------------------------------------------------------------------------
  //   Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, TensorLayoutBank])
  : TensorLayoutBuffer = TensorLayoutBuffer(banks)

  def toParameterBuffer(convertFn: IndependentTensorLayout => RealTensor)
  : ValueTensorBuffer = {
    val result = mapBanks(_.toRealTensorBank(convertFn))
    ValueTensorBuffer(result)
  }

}

object TensorLayoutBuffer {

  final def apply(banks: SortedMap[Int, TensorLayoutBank])
  : TensorLayoutBuffer = new TensorLayoutBuffer(banks)

  final def derive(bankNo: Int, layout: TensorLayoutBank)
  : TensorLayoutBuffer = derive((bankNo, layout))

  final def derive(bank0: (Int, TensorLayoutBank))
  : TensorLayoutBuffer = apply(SortedMap(bank0))

  final def derive(json: JValue)
  : TensorLayoutBuffer = derive(json.asInstanceOf[JObject])

  final def derive(json: JObject)
  : TensorLayoutBuffer = {
    val fields = json.obj.toMap
    val result = Json.toSortedMap(
      fields("banks"),
      (json: JValue) => Json.toInt(json),
      (json: JValue) => TensorLayoutBank.derive(json)
    )
    apply(result)
  }

  final val empty
  : TensorLayoutBuffer = apply(SortedMap.empty)

}

final class TensorLayoutBufferBuilder
  extends BufferExBuilder[TensorLayoutBuffer, TensorLayoutBank, IndependentTensorLayout] {

  override protected def doRegister(bankNo:    Int,
                                    segmentNo: Int,
                                    item:      IndependentTensorLayout)
  : Int = {
    val bank = banks.getOrElseUpdate(bankNo, TensorLayoutBankBuilder())
    bank.register(segmentNo, item)
  }

  override def result()
  : TensorLayoutBuffer = TensorLayoutBuffer(toSortedMap)

}

object TensorLayoutBufferBuilder {

  final def apply()
  : TensorLayoutBufferBuilder = new TensorLayoutBufferBuilder

}
