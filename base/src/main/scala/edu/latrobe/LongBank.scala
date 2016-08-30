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

final class LongBank(override val segments: SortedMap[Int, Long])
  extends BankEx[LongBank, Long] {

  override def toString
  : String = s"LongBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LongBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LongBank =>
      segments == other.segments
    case _ =>
      false
  })

  def +(value: Long)
  : LongBank = {
    val result = mapSegments(
      _ + value
    )
    LongBank(result)
  }

  def +(other: LongBank)
  : LongBank = {
    val result = zipSegmentsEx(
      other
    )((s0, s1) => s0 + s1, s0 => s0, s1 => s1)
    LongBank(result)
  }

  def -(value: Long)
  : LongBank = {
    val result = mapSegments(
      _ - value
    )
    LongBank(result)
  }

  def -(other: LongBank)
  : LongBank = {
    val result = zipSegmentsEx(
      other
    )((s0, s1) => s0 - s1, s0 => s0, s1 => s1)
    LongBank(result)
  }


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, Long])
  : LongBank = LongBank(banks)

  override protected def doToJson(segment: Long)
  : JValue = Json(segment)

}

object LongBank
  extends BankExCompanion[LongBank, Long] {

  final override def apply(segments: SortedMap[Int, Long])
  : LongBank = new LongBank(segments)

  final override protected def doDerive(json: JValue)
  : Long = Json.toLong(json)

  final def fillLike(bank:  BankLike,
                     value: Long)
  : LongBank = {
    val result = MapEx.mapValues(
      bank.segments
    )(s => value)
    apply(result)
  }

  final def zeroLike(bank: BankLike)
  : LongBank = fillLike(bank, 0L)

}
