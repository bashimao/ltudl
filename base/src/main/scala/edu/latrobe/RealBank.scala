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

final class RealBank(override val segments: SortedMap[Int, Real])
  extends BankEx[RealBank, Real] {

  override def toString
  : String = s"RealBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RealBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RealBank =>
      segments == other.segments
    case _ =>
      false
  })


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, Real])
  : RealBank = RealBank(banks)

  override protected def doToJson(segment: Real)
  : JValue = Json(segment)

}

object RealBank
  extends BankExCompanion[RealBank, Real] {

  final override def apply(segments: SortedMap[Int, Real])
  : RealBank = new RealBank(segments)

  final override protected def doDerive(json: JValue)
  : Real = Json.toReal(json)

  final def fillLike(bank:  BankLike,
                     value: Real)
  : RealBank = {
    val result = MapEx.mapValues(
      bank.segments
    )(s => value)
    apply(result)
  }

  final def minusOneLike(bank: BankLike)
  : RealBank = fillLike(bank, -Real.one)

  final def oneLike(bank: BankLike)
  : RealBank = fillLike(bank, Real.one)

  final def zeroLike(bank: BankLike)
  : RealBank = fillLike(bank, Real.zero)

}
