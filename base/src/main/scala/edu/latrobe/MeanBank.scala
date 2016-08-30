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

final class MeanBank(override val segments: SortedMap[Int, Mean])
  extends BankEx[MeanBank, Mean]
    with CopyableEx[MeanBank] {
  require(!segments.exists(_._2 == null))

  override def toString
  : String = s"MeanBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MeanBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MeanBank =>
      segments == other.segments
    case _ =>
      false
  })

  override def copy
  : MeanBank = MeanBank(mapSegments(_.copy))


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, Mean])
  : MeanBank = MeanBank(banks)

  override protected def doToJson(segment: Mean)
  : JObject = segment.toJson

}

object MeanBank
  extends BankExCompanion[MeanBank, Mean] {

  final override def apply(segments: SortedMap[Int, Mean])
  : MeanBank = new MeanBank(segments)

  final override protected def doDerive(json: JValue)
  : Mean = Mean.derive(json)

}
