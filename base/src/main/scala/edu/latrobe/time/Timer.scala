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
import org.json4s._
import scala.collection._
import scala.util.hashing._

/**
  * An inverted stopwatch.
  */
@SerialVersionUID(1L)
final class Timer(private var _end: Timestamp)
  extends ClockEx[Timer] {

  def end
  : Timestamp = _end

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _end.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Timer]

  override def copy
  : Timer = Timer(_end)

  def elapsed
  : Boolean = Timestamp.now() >= _end

  override def read()
  : TimeSpan = {
    val now = Timestamp.now()
    if (now >= _end) {
      TimeSpan.zero
    }
    else {
      TimeSpan(now, _end)
    }
  }

  def readAndReset(end: Timestamp)
  : TimeSpan = {
    val now = Timestamp.now()
    val res = {
      if (now >= _end) {
        TimeSpan.zero
      }
      else {
        TimeSpan(now, _end)
      }
    }
    _end = end
    res
  }

  def readAndReset(duration: TimeSpan)
  : TimeSpan = readAndReset(Timestamp.now() + duration)

  def readAndResetAs(label: String, end: Timestamp)
  : LabeledTimeSpan = LabeledTimeSpan(label, readAndReset(end))

  def readAndResetAs(label: String, duration: TimeSpan)
  : LabeledTimeSpan = LabeledTimeSpan(label, readAndReset(duration))

  def reset(end: Timestamp)
  : Unit = {
    require(end != null)
    _end = end
  }

  def reset(duration: TimeSpan)
  : Unit = reset(Timestamp.now() + duration)

  def resetIfElapsed(duration: TimeSpan)
  : Boolean = {
    val now = Timestamp.now()
    val res = now >= _end
    if (res) {
      reset(now + duration)
    }
    res
  }

  override protected def doToJson()
  : List[JField] = List(
    Json.field("end", Timestamp.toJson(_end))
  )

}

object Timer
  extends JsonSerializableCompanionEx[Timer]{

  final def apply(end: Timestamp)
  : Timer = new Timer(end)

  final def apply(duration: TimeSpan)
  : Timer = apply(Timestamp.now() + duration)

  final override def derive(fields: Map[String, JValue])
  : Timer = apply(Timestamp.derive(fields("end")))

}