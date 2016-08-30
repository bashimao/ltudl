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

package edu.latrobe.io.vega

import edu.latrobe._
import org.json4s.JsonAST._
import scala.util.hashing._

@SerialVersionUID(1L)
final class DataPoint2D(val x: Real,
                        val y: Real)
  extends Serializable
    with JsonSerializable
    with Equatable {

  override def toString
  : String = f"$x%.4g, $y%.4g"

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, x.hashCode())
    tmp = MurmurHash3.mix(tmp, y.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DataPoint2D]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DataPoint2D =>
      x == other.x &&
      y == other.y
    case _ =>
      false
  })

  @inline
  def +(other: DataPoint2D)
  : DataPoint2D = DataPoint2D(
    x + other.x,
    y + other.y
  )

  @inline
  def -(other: DataPoint2D)
  : DataPoint2D = DataPoint2D(
    x - other.x,
    y - other.y
  )

  @inline
  def *(value: Real)
  : DataPoint2D = DataPoint2D(
    x * value,
    y * value
  )

  @inline
  def /(value: Real)
  : DataPoint2D = DataPoint2D(
    x / value,
    y / value
  )

  @inline
  def lerp(other: DataPoint2D, t: Real)
  : DataPoint2D = DataPoint2D(
    MathMacros.lerp(x, other.x, t),
    MathMacros.lerp(y, other.y, t)
  )

  override protected def doToJson()
  : List[JField] = List(
    Json.field("x", x),
    Json.field("y", y)
  )

  def toTuple
  : (Real, Real) = (x, y)

}

object DataPoint2D {

  final def apply(x: Real,
                  y: Real)
  : DataPoint2D = new DataPoint2D(x, y)

  final def derive(json: JValue)
  : DataPoint2D = derive(json.asInstanceOf[JObject])

  final def derive(json: JObject)
  : DataPoint2D = {
    val fields = json.obj.toMap
    DataPoint2D(
      Json.toReal(fields("x")),
      Json.toReal(fields("y"))
    )
  }

  final val minusOne
  : DataPoint2D = apply(-Real.one, -Real.one)

  final val nan
  : DataPoint2D = apply(Real.nan, Real.nan)

  final val one
  : DataPoint2D = apply(Real.one, Real.one)

  final val zero
  : DataPoint2D = apply(Real.zero, Real.zero)

}
