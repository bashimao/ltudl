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

object FloatEx {

  final val nan
  : Float = Float.NaN

  final val negativeInfinity
  : Float = Float.NegativeInfinity

  final val positiveInfinity
  : Float = Float.PositiveInfinity

  final val zero
  : Float = 0.0f

  final val one
  : Float = 1.0f

  final val two
  : Float = 2.0f

  final val pointFive
  : Float = 0.5f

  final val epsilon
  : Float = Float.MinPositiveValue

  // Small number you can divide by safely.
  // (across a large value range without running into numeric issues)
  final val minQuotient
  : Float = 1e-30f

  // Small number that you can divide 1 by, that will not result in infinity.
  final val minQuotient1
  : Float = 0.293873657771e-38f

  final val minValue
  : Float = Float.MinValue

  final val maxValue
  : Float = Float.MaxValue

  final val sizeInBits
  : Int = java.lang.Float.SIZE

  final val size
  : Int = sizeInBits / 8

  @inline
  final def apply(value: Byte)
  : Float = value.toFloat

  @inline
  final def apply(value: Char)
  : Float = value.toFloat

  @inline
  final def apply(value: Int)
  : Float = value.toFloat

  @inline
  final def apply(value: Long)
  : Float = value.toFloat

  @inline
  final def apply(value: Float)
  : Float = value

  @inline
  final def apply(value: Double)
  : Float = value.toFloat

  @inline
  final def apply(value: String)
  : Float = value.toFloat

  @inline
  final def isInfinite(value: Float)
  : Boolean = java.lang.Float.isInfinite(value)

  @inline
  final def isNaN(value: Float)
  : Boolean = java.lang.Float.isNaN(value)

}