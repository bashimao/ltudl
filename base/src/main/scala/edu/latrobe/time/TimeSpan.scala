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
import org.json4s.JsonAST._

object TimeSpan {

  final def apply(seconds: Real)
  : TimeSpan = Duration.millis(Math.round(seconds * 1000.0))

  final def apply(start: ReadableInstant, end: ReadableInstant)
  : TimeSpan = new Duration(start, end)

  final def derive(json: JValue)
  : TimeSpan = Duration.millis(Json.toLong(json))

  final def toJson(value: TimeSpan)
  : JInt = Json(value.getMillis)

  final val zero
  : TimeSpan = Duration.ZERO

  final val oneMillisecond
  : TimeSpan = Duration.millis(1L)

  final val fiveMilliseconds
  : TimeSpan = Duration.millis(5L)

  final val tenMilliseconds
  : TimeSpan = Duration.millis(10L)

  final val hundredMilliseconds
  : TimeSpan = Duration.millis(100L)

  final val fiveHundredMilliseconds
  : TimeSpan = Duration.millis(500L)

  final val oneSecond
  : TimeSpan = Duration.millis(1000L)

  final val threeSeconds
  : TimeSpan = Duration.millis(1000L * 3L)

  final val fiveSeconds
  : TimeSpan = Duration.millis(1000L * 5L)

  final val tenSeconds
  : TimeSpan = Duration.millis(1000L * 10L)

  final val thirtySeconds
  : TimeSpan = Duration.millis(1000L * 30L)

  final val fiftySeconds
  : TimeSpan = Duration.millis(1000L * 50L)

  final val hundredSeconds
  : TimeSpan = Duration.millis(1000L * 100L)

  final val oneMinute
  : TimeSpan = Duration.millis(1000L * 60L)

  final val fiveMinutes
  : TimeSpan = Duration.millis(1000L * 60L * 5L)

  final val tenMinutes
  : TimeSpan = Duration.millis(1000L * 60L * 10L)

  final val hundredMinutes
  : TimeSpan = Duration.millis(1000L * 60L * 100L)

  final val oneHour
  : TimeSpan = Duration.millis(1000L * 60L * 60L)

  final val fiveHours
  : TimeSpan = Duration.millis(1000L * 60L * 60L * 5L)

  final val tenHours
  : TimeSpan = Duration.millis(1000L * 60L * 60L * 10L)

  final val oneDay
  : TimeSpan = Duration.millis(1000L * 60L * 60L * 24L)

  final val threeDays
  : TimeSpan = Duration.millis(1000L * 60L * 60L * 24L * 3L)

  final val fiveDays
  : TimeSpan = Duration.millis(1000L * 60L * 60L * 24L * 5L)

  final val tenDays
  : TimeSpan = Duration.millis(1000L * 60L * 60L * 24L * 10L)

  final val thirtyDays
  : TimeSpan = Duration.millis(1000L * 60L * 60L * 24L * 30L)

  final val infinite
  : TimeSpan = Duration.millis(Long.MaxValue)

}