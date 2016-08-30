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

final class BooleanBank(override val segments: SortedMap[Int, Boolean])
  extends BankEx[BooleanBank, Boolean] {

  override def toString
  : String = s"BooleanBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BooleanBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BooleanBank =>
      segments == other.segments
    case _ =>
      false
  })

  def &&(value: Boolean)
  : BooleanBank = {
    val result = mapSegments(
      _ && value
    )
    BooleanBank(result)
  }

  def ||(value: Boolean)
  : BooleanBank = {
    val result = mapSegments(
      _ || value
    )
    BooleanBank(result)
  }


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, Boolean])
  : BooleanBank = BooleanBank(banks)

  override protected def doToJson(segment: Boolean)
  : JValue = Json(segment)

}

object BooleanBank
  extends BankExCompanion[BooleanBank, Boolean] {

  final override def apply(segments: SortedMap[Int, Boolean])
  : BooleanBank = new BooleanBank(segments)

  final override protected def doDerive(json: JValue)
  : Boolean = Json.toBoolean(json)

  final def deriveTrue(segmentNo: Int)
  : BooleanBank = derive(segmentNo, value = true)

  final def deriveFalse(segmentNo: Int)
  : BooleanBank = derive(segmentNo, value = false)

  final def fillLike(bank:  BankLike,
                     value: Boolean)
  : BooleanBank = {
    val result = MapEx.mapValues(
      bank.segments
    )(s => value)
    apply(result)
  }

  final def trueLike(bank: BankLike)
  : BooleanBank = fillLike(bank, value = true)

  final def falseLike(bank: BankLike)
  : BooleanBank = fillLike(bank, value = false)

}
