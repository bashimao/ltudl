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

final class IntBank(override val segments: SortedMap[Int, Int])
  extends BankEx[IntBank, Int] {

  override def toString
  : String = s"IntBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IntBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: IntBank =>
      segments == other.segments
    case _ =>
      false
  })

  def sum
  : Long = {
    MapEx.foldLeftValues(
      0L,
      segments
    )(_ + _)
  }

  def +(value: Int)
  : IntBank = {
    val result = mapSegments(
      _ + value
    )
    IntBank(result)
  }

  def +(other: IntBank)
  : IntBank = {
    val result = zipSegmentsEx(
      other
    )((s0, s1) => s0 + s1, s0 => s0, s1 => s1)
    IntBank(result)
  }

  def -(value: Int)
  : IntBank = {
    val result = mapSegments(
      _ - value
    )
    IntBank(result)
  }

  def -(other: IntBank)
  : IntBank = {
    val result = zipSegmentsEx(
      other
    )((s0, s1) => s0 - s1, s0 => s0, s1 => s1)
    IntBank(result)
  }


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, Int])
  : IntBank = IntBank(banks)

  override protected def doToJson(segment: Int)
  : JValue = Json(segment)

}

object IntBank
  extends BankExCompanion[IntBank, Int] {

  final override def apply(segments: SortedMap[Int, Int])
  : IntBank = new IntBank(segments)

  final override protected def doDerive(json: JValue)
  : Int = Json.toInt(json)

  final def fillLike(bank:  BankLike,
                     value: Int)
  : IntBank = {
    val result = MapEx.mapValues(
      bank.segments
    )(s => value)
    apply(result)
  }

  final def zeroLike(bank: BankLike)
  : IntBank = fillLike(bank, 0)

}
