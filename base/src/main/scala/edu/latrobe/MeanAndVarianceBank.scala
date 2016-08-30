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

final class MeanAndVarianceBank(override val segments: SortedMap[Int, MeanAndVariance])
  extends BankEx[MeanAndVarianceBank, MeanAndVariance]
    with CopyableEx[MeanAndVarianceBank] {
  require(!segments.exists(_._2 == null))

  override def toString
  : String = s"MeanAndVarianceBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MeanAndVarianceBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MeanAndVarianceBank =>
      segments == other.segments
    case _ =>
      false
  })

  override def copy
  : MeanAndVarianceBank = {
    val result = mapSegments(_.copy)
    MeanAndVarianceBank(result)
  }


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, MeanAndVariance])
  : MeanAndVarianceBank = MeanAndVarianceBank(banks)

  override protected def doToJson(segment: MeanAndVariance)
  : JObject = segment.toJson

}

object MeanAndVarianceBank
  extends BankExCompanion[MeanAndVarianceBank, MeanAndVariance] {

  final override def apply(segments: SortedMap[Int, MeanAndVariance])
  : MeanAndVarianceBank = new MeanAndVarianceBank(segments)

  final override protected def doDerive(json: JValue)
  : MeanAndVariance = MeanAndVariance.derive(json)

}
