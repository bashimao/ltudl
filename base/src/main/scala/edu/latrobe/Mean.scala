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

import scala.collection.Map
import scala.util.hashing._

trait MeanLike
  extends MutableAccumulatorLike
    with Equatable
    with Serializable
    with JsonSerializable {

  final protected var _count
  : Long = 0L

  final protected var _mean
  : Double = 0.0

  final def mean
  : Real = Real(_mean)

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _mean.hashCode)

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MeanLike =>
      _count == other._count &&
      _mean  == other._mean
    case _ =>
      false
  })

  override def reset()
  : Unit = {
    _count = 0L
    _mean  = 0.0
  }

  final def reset(json: JObject)
  : Unit = doReset(json.obj.toMap)

  protected def doReset(fields: Map[String, JValue])
  : Unit = {
    _count = Json.toInt(fields("count"))
    _mean  = Json.toDouble(fields("mean"))
  }

}

abstract class MeanLikeCompanion
  extends JsonSerializableCompanion {
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
  */
final class Mean
  extends MeanLike
    with MutableAccumulatorLikeEx[Mean]
    with Comparable[Mean] {

  override def toString
  : String = f"mu=$mean%.4g"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Mean]

  override def compareTo(o: Mean)
  : Int = _mean.compareTo(o._mean)

  override def copy
  : Mean = {
    val res = Mean()
    res := this
    res
  }

  @inline
  override def update(value: Real)
  : Unit = {
    _count += 1L
    _mean  += (value - _mean) / _count
  }

  override def :=(other: Mean)
  : Unit = {
    _count = other._count
    _mean  = other._mean
  }

  override def +=(other: Mean)
  : Unit = {
    _count = _count + other._count
    val t  = other._count.toDouble / _count.toDouble
    _mean  = MathMacros.lerp(_mean, other._mean, t)
  }

  override protected def doToJson()
  : List[JField] = List(
    Json.field("count", _count),
    Json.field("mean",  _mean)
  )

}

object Mean
  extends MeanLikeCompanion
    with MutableAccumulatorLikeExCompanion[Mean]
    with JsonSerializableCompanionEx[Mean] {

  final def apply()
  : Mean = new Mean

  final override def derive(fields: Map[String, JValue])
  : Mean = {
    val result = apply()
    result.doReset(fields)
    result
  }

}
