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

final class SimpleBufferReference private(override val bankNo:    Int,
                                          override val segmentNo: Int)
  extends BufferReferenceEx[SimpleBufferReference] {

  override def toString
  : String = s"$bankNo/$segmentNo"

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, bankNo.hashCode())
    tmp = MurmurHash3.mix(tmp, segmentNo.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SimpleBufferReference]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SimpleBufferReference =>
      bankNo    == other.bankNo &&
      segmentNo == other.segmentNo
    case _ =>
      false
  })

  override def derive(segmentNo: Int)
  : SimpleBufferReference = SimpleBufferReference(bankNo, segmentNo)

  def +(other: SimpleBufferReference)
  : NullBuffer = NullBuffer.derive(this, other)


  // ---------------------------------------------------------------------------
  //    Conversion related.
  // ---------------------------------------------------------------------------
  override protected def doToJson()
  : List[JField] = List(
    Json.field("bankNo",    bankNo),
    Json.field("segmentNo", segmentNo)
  )

  def toLabeledBufferReference(handle: String)
  : LabeledBufferReference = LabeledBufferReference(bankNo, segmentNo, handle)

}

object SimpleBufferReference
  extends JsonSerializableCompanionEx[SimpleBufferReference] {

  final def apply(bankNo:    Int,
                  segmentNo: Int)
  : SimpleBufferReference = new SimpleBufferReference(bankNo, segmentNo)

  final override def derive(fields: Map[String, JValue])
  : SimpleBufferReference = apply(
    Json.toInt(fields("bankNo")),
    Json.toInt(fields("segmentNo"))
  )

  final val zero
  : SimpleBufferReference = apply(0, 0)

}
