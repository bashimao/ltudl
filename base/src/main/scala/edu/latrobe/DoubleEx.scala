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

object DoubleEx {

  final val nan: Double = Double.NaN

  final val negativeInfinity: Double = Double.NegativeInfinity

  final val positiveInfinity: Double = Double.PositiveInfinity

  final val zero: Double = 0.0

  final val one: Double = 1.0

  final val two: Double = 2.0

  final val pointFive: Double = 0.5

  final val epsilon: Double = Double.MinPositiveValue

  // Small number you can divide by safely.
  // (across a large value range without running into numeric issues)
  final val minQuotient
  : Double = 1e-250

  // Small number that you can divide 1 by, that will not result in infinity.
  final val minQuotient1
  : Double = 0.5562684646269e-308

  final val minValue
  : Double = Double.MinValue

  final val maxValue
  : Double = Double.MaxValue

  final val size: Int = 8

  @inline
  final def apply(value: Byte): Double = value.toDouble

  @inline
  final def apply(value: Char): Double = value.toDouble

  @inline
  final def apply(value: Int): Double = value.toDouble

  @inline
  final def apply(value: Long): Double = value.toDouble

  @inline
  final def apply(value: Float): Double = value.toDouble

  @inline
  final def apply(value: Double): Double = value

  @inline
  final def apply(value: String): Double = value.toDouble

  @inline
  final def isInfinite(value: Double)
  : Boolean = java.lang.Double.isInfinite(value)

  @inline
  final def isNaN(value: Double)
  : Boolean = java.lang.Double.isNaN(value)

}
