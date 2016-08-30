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

import org.joda.time._

package object time {

  type Timestamp = Instant

  type TimeSpan = Duration

  /**
    * Joda DateTime type.
    */
  final implicit class JodaReadableInstantFunctions(ri: ReadableInstant) {

    @inline
    def <(other: ReadableInstant)
    : Boolean = ri.isBefore(other)

    @inline
    def <=(other: ReadableInstant)
    : Boolean = !ri.isAfter(other)

    @inline
    def >(other: ReadableInstant)
    : Boolean = ri.isAfter(other)

    @inline
    def >=(other: ReadableInstant)
    : Boolean = !ri.isBefore(other)

  }

  final implicit class JodaTimestampFunctions(i: Timestamp) {

    @inline
    def +(other: ReadableDuration)
    : Instant = i.plus(other)

    @inline
    def -(other: ReadableDuration)
    : Instant = i.minus(other)

  }

  /**
    * Jode Duration type.
    */
  final implicit class JodaReadableDurationFunctions(rd: ReadableDuration) {

    @inline
    def milliseconds
    : Long = rd.getMillis

    @inline
    def seconds
    : Real = rd.getMillis / 1000.0f

    @inline
    def <(other: ReadableDuration)
    : Boolean = rd.isShorterThan(other)

    @inline
    def <=(other: ReadableDuration)
    : Boolean = !rd.isLongerThan(other)

    @inline
    def >(other: ReadableDuration)
    : Boolean = rd.isLongerThan(other)

    @inline
    def >=(other: ReadableDuration)
    : Boolean = !rd.isShorterThan(other)

  }

  final implicit class JodaTimeSpanFunctions(ts: TimeSpan) {

    @inline
    def +(other: ReadableDuration)
    : Duration = ts.plus(other)

    @inline
    def -(other: ReadableDuration)
    : Duration = ts.minus(other)

    @inline
    def *(other: Long)
    : Duration = ts.multipliedBy(other)

    @inline
    def /(other: Long)
    : Duration = ts.dividedBy(other)

  }

}
