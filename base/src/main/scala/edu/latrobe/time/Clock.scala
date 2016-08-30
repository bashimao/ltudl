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

/**
  * A mutable object for measuring time.
  */
abstract class Clock
  extends Equatable
    with Comparable[Clock]
    with Copyable
    with Serializable
    with JsonSerializable {

  final override def toString
  : String = f"${read().seconds}%.3f s"

  final override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Clock =>
      read() == other.read()
    case _ =>
      false
  })

  final override def compareTo(other: Clock)
  : Int = read().compareTo(other.read())

  def read()
  : TimeSpan

  final def readAs(label: String)
  : LabeledTimeSpan = LabeledTimeSpan(label, read())

}

abstract class ClockEx[TThis <: ClockEx[_]]
  extends Clock
    with CopyableEx[TThis] {

  final def compareTo(other: TimeSpan)
  : Int = {
    val state = read()
    state.compareTo(other)
  }

  final def >(other: Clock)
  : Boolean = compareTo(other) > 0

  final def >(other: TimeSpan)
  : Boolean = compareTo(other) > 0

  final def >=(other: Stopwatch)
  : Boolean = compareTo(other) >= 0

  final def >=(other: TimeSpan)
  : Boolean = compareTo(other) >= 0

  final def <(other: Stopwatch)
  : Boolean = compareTo(other) < 0

  final def <(other: TimeSpan)
  : Boolean = compareTo(other) < 0

  final def <=(other: Stopwatch)
  : Boolean = compareTo(other) <= 0

  final def <=(other: TimeSpan)
  : Boolean = compareTo(other) <= 0

}