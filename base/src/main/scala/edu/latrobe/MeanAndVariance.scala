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

trait MeanAndVarianceLike
  extends MeanLike {

  final protected var _varianceAcc
  : Double = 0.0

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _varianceAcc.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MeanAndVarianceLike =>
      _varianceAcc == other._varianceAcc
    case _ =>
      false
  })

  protected def _populationVariance
  : Double

  final def populationVariance
  : Real = Real(_populationVariance)

  @inline
  final private def _populationStdDev(epsilon: Double)
  : Double = Math.sqrt(_populationVariance + epsilon)

  final def populationStdDev()
  : Real = populationStdDev(0.0)

  final def populationStdDev(epsilon: Double)
  : Real = Real(_populationStdDev(epsilon))

  final def populationStdDevInv()
  : Real = populationStdDevInv(0.0)

  final def populationStdDevInv(epsilon: Double)
  : Real = Real(1.0 / _populationStdDev(epsilon))

  protected def _sampleVariance: Double

  final def sampleVariance
  : Real = Real(_sampleVariance)

  @inline
  final private def _sampleStdDev(epsilon: Double)
  : Double = Math.sqrt(_sampleVariance + epsilon)

  final def sampleStdDev()
  : Real = sampleStdDev(0.0)

  final def sampleStdDev(epsilon: Double)
  : Real = Real(_sampleStdDev(epsilon))

  final def sampleStdDevInv()
  : Real = sampleStdDevInv(0.0)

  final def sampleStdDevInv(epsilon: Double)
  : Real = Real(1.0 / _sampleStdDev(epsilon))

  override def reset()
  : Unit = {
    super.reset()
    _varianceAcc = 0.0
  }

  override protected def doReset(fields: Map[String, JValue])
  : Unit = {
    super.doReset(fields)
    _varianceAcc = Json.toDouble(fields("varianceAcc"))
  }

}

abstract class MeanAndVarianceLikeCompanion
  extends MeanLikeCompanion {
}

/**
  * Incremental computation.
  * (see http://www.heikohoffmann.de/htmlthesis/node134.html)
  *
  * mu  = 0
  *   0
  *       i - 1         1              1 (            )
  * mu  = ----- mu    + - x  = mu    + - ( x  - mu    )
  *   i     i     i-1   i  i     i-1   i (  i     i-1 )
  *
  * q  = 0
  *  0
  *                                 2
  *             i - 1 (            )                   1 (           ) (           )
  * q  = q    + ----- ( x  - mu    )  = q    + (i - 1) - ( x  - m    ) ( x  - m    )
  *   i   i-1     i   (  i     i-1 )     i-1           i (  i    i-1 ) (  i    i-1 )
  *
  * Sample variance:
  *            q
  *      2      n
  * sigma  = -----
  *      n   n - 1
  *
  * Population variance:
  *             q
  *        2     n
  * p_sigma  = ---
  *        n    n
  *
  */
final class MeanAndVariance
  extends MeanAndVarianceLike
    with MutableAccumulatorLikeEx[MeanAndVariance] {

  override def toString
  : String = f"mu=$mean%.4g, sigma=${populationStdDev()}%.4g"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MeanAndVariance]

  override def copy
  : MeanAndVariance = {
    val res = MeanAndVariance()
    res := this
    res
  }

  override protected def _populationVariance
  : Double = if (_count > 0L) _varianceAcc / _count else 0.0

  override protected def _sampleVariance
  : Double = if (_count > 1L) _varianceAcc / (_count - 1L) else 0.0


  @inline
  override def update(value: Real)
  : Unit = {
    val count1  = _count + 1L
    val diff    = value - _mean
    val diffByN = diff / count1

    _mean        += diffByN
    _varianceAcc += _count * diffByN * diff

    _count = count1
  }

  override def :=(other: MeanAndVariance)
  : Unit = {
    _count       = other._count
    _mean        = other._mean
    _varianceAcc = other._varianceAcc
  }

  override def +=(other: MeanAndVariance)
  : Unit = {
    _count       = _count + other._count
    val t        = other._count.toDouble / _count.toDouble
    _mean        = MathMacros.lerp(_mean,        other._mean,        t)
    _varianceAcc = MathMacros.lerp(_varianceAcc, other._varianceAcc, t)
  }

  override protected def doToJson()
  : List[JField] = List(
    Json.field("count",       _count),
    Json.field("mean",        _mean),
    Json.field("varianceAcc", _varianceAcc)
  )
}

object MeanAndVariance
  extends MeanAndVarianceLikeCompanion
    with MutableAccumulatorLikeExCompanion[MeanAndVariance]
    with JsonSerializableCompanionEx[MeanAndVariance] {

  final def apply()
  : MeanAndVariance = new MeanAndVariance

  override def derive(fields: Map[String, JValue])
  : MeanAndVariance = {
    val result = apply()
    result.doReset(fields)
    result
  }

}
