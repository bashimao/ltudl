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

import scala.util.hashing._

/**
  * Simpler version of Range. That has less overhead than
  * org.apache.commons.lang3.Range
  * and does less checks than
  * scala.collection.immutable.NumericRange.
  * @param min Inclusive minimum value.
  * @param max Inclusive maximum value.
  */
final class RealRange(val min: Real,
                      val max: Real)
  extends Serializable
    with Equatable
    with Cloneable {
  require(min <= max)

  override def toString
  : String = f"[$min%.4g, $max%.4g]"

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, min.hashCode())
    tmp = MurmurHash3.mix(tmp, max.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RealRange]

  /**
    * Remark: Make sure this one is versatile!
    */
  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RealRange =>
      min == other.min &&
      max == other.max
    case _ =>
      false
  })

  override def clone()
  : RealRange = RealRange(min, max)

  @inline
  def contains(value: Real)
  : Boolean = value >= min && value <= max

  @inline
  def clip(value: Real)
  : Real = {
    if (value < min) {
      return min
    }
    if (value > max) {
      return max
    }
    value
  }

  @inline
  def clipAndWarn(value: Real, valueName: String = "value")
  : Real = {
    val result = clip(value)
    if (result != value) {
      logger.warn(
        f"Supplied $valueName was out of bounds [$min%.4g, $max%.4g] and clipped: $value%.4g => $result%.4g!"
      )
    }
    result
  }

  @inline
  def length
  : Real = max - min

  @inline
  def isInfinite
  : Boolean = Real.isInfinite(min) || Real.isInfinite(max)

  @inline
  def isNaN
  : Boolean = Real.isNaN(min) || Real.isNaN(max)

}

/**
  * Ranges always inclusive!
  */
object RealRange {

  final def apply(min: Real,
                  max: Real)
  : RealRange = new RealRange(min, max)

  final def derive(max: Real)
  : RealRange = new RealRange(-max, max)

  /**
    * [0, 0]
    */
  final val zero
  : RealRange = apply(Real.zero, Real.zero)

  /**
    * [0, 1]
    */
  final val zeroToOne
  : RealRange = apply(Real.zero, Real.one)

  /**
    * [-1, 1]
    */
  final val minusOneToOne
  : RealRange = apply(-Real.one, Real.one)

  /**
    * [-inf, inf]
    */
  final val infinite
  : RealRange = apply(Real.negativeInfinity, Real.positiveInfinity)

  /**
    * [0, inf]
    */
  final val zeroToInfinity
  : RealRange = apply(Real.zero, Real.positiveInfinity)

  /**
    * [1, inf]
    */
  final val oneToInfinity
  : RealRange = apply(Real.one, Real.positiveInfinity)

}
