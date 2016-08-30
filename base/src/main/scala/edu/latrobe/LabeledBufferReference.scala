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

final class LabeledBufferReference private[latrobe](override val bankNo:    Int,
                                                    override val segmentNo: Int,
                                                    val          handle:    String)
  extends BufferReferenceEx[LabeledBufferReference] {

  override def toString
  : String = s"$bankNo/$segmentNo = $handle"

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, bankNo.hashCode())
    tmp = MurmurHash3.mix(tmp, segmentNo.hashCode())
    tmp = MurmurHash3.mix(tmp, handle.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LabeledBufferReference]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LabeledBufferReference =>
      bankNo    == other.bankNo    &&
      segmentNo == other.segmentNo &&
      handle    == other.handle
    case _ =>
      false
  })

  def derive(handle: String)
  : LabeledBufferReference = LabeledBufferReference(bankNo, segmentNo, handle)

  // TODO: Figure out how to protect against us setting a bad segment number except for the one case where we want that.
  override def derive(segmentNo: Int)
  : LabeledBufferReference = {
    // Note that this does not check the segmentNo.
    new LabeledBufferReference(bankNo, segmentNo, handle)
  }


  // ---------------------------------------------------------------------------
  //    Conversion related.
  // ---------------------------------------------------------------------------
  override protected def doToJson()
  : List[JField] = List(
    Json.field("bankNo",    bankNo),
    Json.field("segmentNo", segmentNo),
    Json.field("handle",    handle)
  )

  def toBufferReference
  : SimpleBufferReference = SimpleBufferReference(bankNo, segmentNo)

}

object LabeledBufferReference
  extends JsonSerializableCompanionEx[LabeledBufferReference] {

  final def apply(bankNo:    Int,
                  segmentNo: Int,
                  handle:    String)
  : LabeledBufferReference = {
    require(segmentNo >= 0 && bankNo >= 0 && handle != null)
    new LabeledBufferReference(bankNo, segmentNo, handle)
  }

  final def apply(handle: String)
  : LabeledBufferReference = apply(0, 0, handle)

  final override def derive(fields: Map[String, JValue])
  : LabeledBufferReference = apply(
    Json.toInt(fields("bankNo")),
    Json.toInt(fields("segmentNo")),
    Json.toString(fields("handle"))
  )

}