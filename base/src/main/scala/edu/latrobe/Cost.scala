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

/**
 * Raw cost tends to overflow. To avoid this we always store the mean cost.
 */
final class Cost(val value:     Real,
                 val noSamples: Long)
  extends Equatable
    with Serializable
    with JsonSerializable {

  override def toString
  : String = f"Cost[$value%.4g, $noSamples%d]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Cost]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, value.hashCode())
    tmp = MurmurHash3.mix(tmp, noSamples.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Cost =>
      value     == other.value &&
      noSamples == other.noSamples
    case _ =>
      false
  })

  /**
   * The raw cost of all covered items (cost * noSamples)!
   *
   * @return
   */
  def rawValue
  : Real = value * noSamples

  def +(amount: Real)
  : Cost = Cost(value + amount, noSamples)

  def +(other: Cost)
  : Cost = {
    val n = noSamples + other.noSamples
    val v = {
      if (n == noSamples) {
        Real.zero
      }
      else {
        MathMacros.lerp(
          value,
          other.value,
          other.noSamples / Real(n)
        )
      }
    }
    Cost(v, n)
  }

  def *(factor: Real)
  : Cost = Cost(value * factor, noSamples)

  override protected def doToJson()
  : List[JField] = List(
    Json.field("value",     value),
    Json.field("noSamples", noSamples)
  )

}

object Cost
  extends JsonSerializableCompanionEx[Cost] {

  final def apply(value:     Real,
                  noSamples: Long)
  : Cost = new Cost(value, noSamples)

  final override def derive(fields: Map[String, JValue])
  : Cost = Cost(
    Json.toReal(fields("value")),
    Json.toLong(fields("noSamples"))
  )

  final val nan
  : Cost = apply(Real.nan, 0L)

  final val zero
  : Cost = apply(Real.zero, 0L)

}
