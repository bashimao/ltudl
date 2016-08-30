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

trait WeightedMeanAndVarianceLike
  extends MeanAndVarianceLike {

  final protected var _weightAcc
  : Double = 0.0

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _weightAcc.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: WeightedMeanAndVarianceLike =>
      _weightAcc == other._weightAcc
    case _ =>
      false
  })

  override def reset()
  : Unit = {
    super.reset()
    _weightAcc = 0.0
  }

  override protected def doReset(fields: Map[String, JValue])
  : Unit = {
    super.doReset(fields)
    _weightAcc = Json.toDouble(fields("weightAcc"))
  }

}

abstract class WeightedMeanAndVarianceLikeCompanion
  extends MeanAndVarianceLikeCompanion {
}

/**
  * Incremental computation.
  * (see http://www.heikohoffmann.de/htmlthesis/node134.html)
  *
  * W  = 0
  *  0
  *
  * W  = W    + w
  *  i    i-1    i
  *
  *       W            w               w
  *        i-1          i               i (            )
  * mu  = ---- mu    + -- x  = mu    + -- ( x  - mu    )
  *   i    W     i-1   W   i     i-1   W  (  i     i-1 )
  *         i           i               i
  *
  *             w  W                  2                 w
  *              i  i-1 (            )                   i (            ) (            )
  * q  = q    + ------- ( x  - mu    )  = q    + W    * -- ( x  - mu    ) ( x  - mu    )
  *  i    i-1      W    (  i     i-1 )     i-1    i-1   W  (  i     i-1 ) (  i     i-1 )
  *                 i                                    i
  *
  * n' = Number of non-zero weights. Hence, zero weighted elements are stenciled
  *      out if this is used. This may not necessarily what somebody wants.
  *
  * Sample variance:
  *
  *      2     n'          2
  * sigma  = ------ p_sigma
  *      n   n' - 1        n
  *
  * Population variance:
  *
  *            q
  *        2    n
  * p_sigma  = --
  *        n   W
  *             n
  *
  */
final class WeightedMeanAndVariance
  extends WeightedMeanAndVarianceLike
    with MutableAccumulatorLikeEx[WeightedMeanAndVariance] {

  override def toString
  : String = f"mu=$mean%.4g, sigma=${populationStdDev()}%.4g"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[WeightedMeanAndVariance]

  override def copy
  : WeightedMeanAndVariance = {
    val res = WeightedMeanAndVariance()
    res := this
    res
  }

  override protected def _populationVariance
  : Double = if (_weightAcc > 0.0) _varianceAcc / _weightAcc else 0.0

  override protected def _sampleVariance
  : Double = {
    if (_count > 1L) {
      _count.toDouble * (_count - 1L) * _populationVariance
    }
    else {
      0.0
    }
  }

  @inline
  override def update(value: Real)
  : Unit = update(value, Real.one)

  @inline
  def update(value: Real, weight: Real)
  : Unit = update(value, weight, Real.epsilon)

  @inline
  def update(value: Real, weight: Real, stencilThreshold: Real)
  : Unit = {
    if (weight < Real.zero) {
      throw new IllegalArgumentException
    }
    else if (weight >= stencilThreshold) {
      _count += 1L
    }

    val weightAcc1   = _weightAcc + weight
    val diff         = value - _mean
    val weightedDiff = weight / weightAcc1 * diff

    _mean        += weightedDiff
    _varianceAcc += _weightAcc * weightedDiff * diff

    _weightAcc = weightAcc1
  }

  override protected def doToJson()
  : List[JField] = List(
    Json.field("count",       _count),
    Json.field("mean",        _mean),
    Json.field("varianceAcc", _varianceAcc),
    Json.field("weightAcc",   _weightAcc)
  )

  override def :=(other: WeightedMeanAndVariance)
  : Unit = {
    _count       = other._count
    _mean        = other._mean
    _varianceAcc = other._varianceAcc
    _weightAcc   = other._weightAcc
  }

  override def +=(other: WeightedMeanAndVariance)
  : Unit = {
    _count       = _count     + other._count
    _weightAcc   = _weightAcc + other._weightAcc
    val t        = other._weightAcc / _weightAcc
    _mean        = MathMacros.lerp(_mean,        other._mean,        t)
    _varianceAcc = MathMacros.lerp(_varianceAcc, other._varianceAcc, t)
  }

}

object WeightedMeanAndVariance
  extends WeightedMeanAndVarianceLikeCompanion
    with JsonSerializableCompanionEx[WeightedMeanAndVariance] {

  final def apply()
  : WeightedMeanAndVariance = new WeightedMeanAndVariance

  final override def derive(fields: Map[String, JValue])
  : WeightedMeanAndVariance = {
    val result = apply()
    result.doReset(fields)
    result
  }

}
