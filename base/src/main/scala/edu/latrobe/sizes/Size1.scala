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

package edu.latrobe.sizes

import edu.latrobe._
import org.json4s.JsonAST._
import scala.collection._
import scala.util.hashing._

final class Size1(override val noTuples:   Int,
                  override val noChannels: Int)
  extends SizeEx[Size1]
    with CartesianSize[Size1, Tuple1[Int]] {
  require(
    noTuples   >= 0 &&
    noChannels >= 0
  )

  override def toString
  : String = s"Size1[$noTuples x $noChannels]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Size1]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, noTuples.hashCode())
    tmp = MurmurHash3.mix(tmp, noChannels.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Size1 =>
      noTuples   == other.noTuples &&
      noChannels == other.noChannels
    case _ =>
      false
  })

  override val noValues
  : Int = noTuples * noChannels

  override def multiplex
  : Size1 = {
    require(noChannels == 1)
    Size1(1, noTuples)
  }

  override def demultiplex
  : Size2 = Size2(noTuples, noChannels, 1)

  override def dims
  : Tuple1[Int] = Tuple1(noTuples)

  override def ++(other: Size)
  : Size1 = {
    require(noChannels == other.noChannels)
    Size1(noTuples + other.noTuples, noChannels)
  }

  override def :++(other: Size)
  : Size1 = {
    require(other match {
      case other: Size1 =>
        noTuples == other.noTuples
      case other: Size2 =>
        other.dims._1 == noTuples &&
        other.dims._2 == 1
      case other: Size3 =>
        other.dims._1 == noTuples &&
        other.dims._2 == 1 &&
        other.dims._3 == 1
      case other: Size4 =>
        other.dims._1 == noTuples &&
        other.dims._2 == 1 &&
        other.dims._3 == 1 &&
        other.dims._4 == 1
      case _ =>
        false
    })
    Size1(noTuples, noChannels + other.noChannels)
  }

  override def withNoTuples(noTuples: Int)
  : Size1 = Size1(noTuples, noChannels)

  override def withNoChannels(noChannels: Int)
  : Size1 = Size1(noTuples, noChannels)


  override protected def doToJson()
  : List[JField] = List(
    Json.field("className",  "Size1"),
    Json.field("noTuples",   noTuples),
    Json.field("noChannels", noChannels)
  )

}

object Size1
  extends SizeExCompanion[Size1] {

  final def apply(noTuples: Int, noChannels: Int)
  : Size1 = new Size1(noTuples, noChannels)

  final def apply(size: Tuple1[Int], noChannels: Int)
  : Size1 = apply(size._1, noChannels)

  final override def derive(fields: Map[String, JValue])
  : Size1 = apply(
    Json.toInt(fields("noTuples")),
    Json.toInt(fields("noChannels"))
  )

  final val one
  : Size1 = apply(1, 1)

  final val zero
  : Size1 = apply(0, 0)

}
