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

@SerialVersionUID(1L)
final class Stopwatch(private var _begin: Timestamp)
  extends ClockEx[Stopwatch] {

  def begin
  : Timestamp = _begin

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _begin.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Stopwatch]

  override def copy
  : Stopwatch = Stopwatch(_begin)

  /*
  def check(reference:    TimeSpan,
            resetIfTrue:  Boolean = false,
            resetIfFalse: Boolean = false)
  : Boolean = {
    val now = Timestamp.now()
    if (TimeSpan(_begin, now) >= reference) {
      if (resetIfTrue) {
        _begin = now
      }
      true
    }
    else {
      if (resetIfFalse) {
        _begin = now
      }
      false
    }
  }

  def checkAndReset(reference: TimeSpan)
  : Boolean = check(reference, resetIfTrue = true, resetIfFalse = true)
  */

  override def read()
  : TimeSpan = TimeSpan(_begin, Timestamp.now())

  def readAndReset()
  : TimeSpan = {
    val now = Timestamp.now()
    val result = TimeSpan(_begin, now)
    _begin = now
    result
  }

  def readAndResetAs(label: String)
  : LabeledTimeSpan = LabeledTimeSpan(label, readAndReset())

  def reset()
  : Unit = _begin = Timestamp.now()

  def reset(begin: Timestamp)
  : Unit = {
    require(begin != null)
    _begin = begin
  }

  def resetIf(predicateFn: TimeSpan => Boolean)
  : Boolean = {
    val now    = Timestamp.now()
    val state  = TimeSpan(_begin, now)
    val result = predicateFn(state)
    if (result) {
      _begin = now
    }
    result
  }

  override protected def doToJson()
  : List[JField] = List(
    Json.field("begin", Timestamp.toJson(_begin))
  )

}

object Stopwatch
  extends JsonSerializableCompanionEx[Stopwatch] {

  final def apply()
  : Stopwatch = apply(Timestamp.now())

  final def apply(begin: Timestamp)
  : Stopwatch = new Stopwatch(begin)

  final override def derive(fields: Map[String, JValue])
  : Stopwatch = apply(Timestamp.derive(fields("begin")))

}