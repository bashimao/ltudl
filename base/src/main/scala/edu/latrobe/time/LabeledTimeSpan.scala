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

package edu.latrobe.time

import edu.latrobe._
import org.joda.time._
import org.json4s.JField
import org.json4s.JsonAST.JValue
import scala.collection._
import scala.util.hashing._

@SerialVersionUID(1L)
final class LabeledTimeSpan(val label: String,
                            val value: TimeSpan)
  extends ReadableDuration
    with Equatable
    with Comparable[ReadableDuration]
    with Serializable
    with JsonSerializable {

  override def toString
  : String = f"$label%s: ${value.seconds}%.3f"

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, label.hashCode())
    tmp = MurmurHash3.mix(tmp, value.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LabeledTimeSpan]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LabeledTimeSpan =>
      label == other.label &&
      value == other.value
    case _ =>
      false
  })

  override def compareTo(other: ReadableDuration)
  : Int = value.compareTo(other)

  override def getMillis
  : Long = value.getMillis

  override def isEqual(other: ReadableDuration)
  : Boolean = value.isEqual(other)

  override def isShorterThan(other: ReadableDuration)
  : Boolean = value.isShorterThan(other)

  override def isLongerThan(other: ReadableDuration)
  : Boolean = value.isLongerThan(other)

  override def toDuration
  : TimeSpan = value

  override def toPeriod
  : Period = value.toPeriod

  def +(other: ReadableDuration)
  : TimeSpan = value.plus(other)

  def -(other: ReadableDuration)
  : TimeSpan = value.minus(other)

  def *(other: Long)
  : TimeSpan = value.multipliedBy(other)

  def /(other: Long)
  : TimeSpan = value.dividedBy(other)

  override protected def doToJson()
  : List[JField] = List(
    Json.field("label", label),
    Json.field("value", TimeSpan.toJson(value))
  )

}

object LabeledTimeSpan
  extends JsonSerializableCompanionEx[LabeledTimeSpan] {

  final def apply(label: String,
                  value: TimeSpan)
  : LabeledTimeSpan = new LabeledTimeSpan(label, value)

  final def derive(label: String,
                   value: Real)
  : LabeledTimeSpan = apply(label, TimeSpan(value))

  final override def derive(fields: Map[String, JValue])
  : LabeledTimeSpan = apply(
    Json.toString(fields("label")),
    TimeSpan.derive(fields("value"))
  )

}
