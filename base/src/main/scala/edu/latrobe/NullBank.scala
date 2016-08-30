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

final class NullBank(override val segments: SortedMap[Int, Null])
  extends BankEx[NullBank, Null] {

  override def toString
  : String = s"NullBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LongBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: NullBank =>
      segments == other.segments
    case _ =>
      false
  })


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, Null])
  : NullBank = NullBank(banks)

  override protected def doToJson(segment: Null)
  : JValue = Json(segment)

}

object NullBank
  extends BankExCompanion[NullBank, Null] {

  final override def apply(segments: SortedMap[Int, Null])
  : NullBank = new NullBank(segments)

  override protected def doDerive(json: JValue)
  : Null = Json.toNull(json)

  final def derive(segmentNo: Int)
  : NullBank = derive((segmentNo, null))

  final def derive[T](segments: SortedMap[Int, T])
  : NullBank = {
    val result = MapEx.mapValues(
      segments
    )(s => null)
    apply(result)
  }

  final def derive(segments: TraversableOnce[Int])
  : NullBank = {
    val builder = SortedMap.newBuilder[Int, Null]
    segments.foreach(
      builder += Tuple2(_, null)
    )
    apply(builder.result())
  }

  final def derive(bank: BankLike)
  : NullBank = {
    val result = MapEx.mapValues(
      bank.segments
    )(s => null)
    apply(result)
  }

}

final class NullBankBuilder
  extends BankExBuilder[NullBank, Null] {

  override def result()
  : NullBank = NullBank(toSortedMap)

}

object NullBankBuilder {

  final def apply()
  : NullBankBuilder = new NullBankBuilder

}
