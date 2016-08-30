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

final class Size2(override val dims:       (Int, Int),
                  override val noChannels: Int)
  extends SizeEx[Size2]
    with CartesianSize[Size2, (Int, Int)] {
  require(
    dims._1    >= 0 &&
    dims._2    >= 0 &&
    noChannels >= 0
  )

  override def toString
  : String = s"Size2[$dims x $noChannels]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Size2]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, dims.hashCode())
    tmp = MurmurHash3.mix(tmp, noChannels.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Size2 =>
      dims       == other.dims &&
      noChannels == other.noChannels
    case _ =>
      false
  })

  override val noTuples
  : Int = dims._1 * dims._2

  override val noValues
  : Int = noTuples * noChannels

  override def multiplex
  : Size1 = {
    require(noChannels == 1)
    Size1(dims._1, dims._2)
  }

  override def demultiplex
  : Size3 = Size3(dims._1, dims._2, noChannels, 1)

  override def ++(other: Size)
  : Size2 = {
    require(noChannels == other.noChannels)
    other match {
      case other: Size1 =>
        require(dims._1 == other.noTuples)
        Size2(
          dims._1,
          dims._2 + 1,
          noChannels
        )
      case other: Size2 =>
        require(dims._1 == other.dims._1)
        Size2(
          dims._1,
          dims._2 + other.dims._2,
          noChannels
        )
      case other: Size3 =>
        require(dims._1 == other.dims._1)
        Size2(
          dims._1,
          dims._2 + (other.dims._2 * other.dims._3),
          noChannels
        )
      case other: Size4 =>
        require(dims._1 == other.dims._1)
        Size2(
          dims._1,
          dims._2 + (other.dims._2 * other.dims._3 * other.dims._4),
          noChannels
        )
      case _ =>
        throw new MatchError(other)
    }
  }

  /**
    * Concatenates each tuple. (Must be of equivalent size!)
    */
  override def :++(other: Size)
  : Size2 = {
    require(other match {
      case other: Size1 =>
        dims._1 == other.noTuples &&
        dims._2 == 1
      case other: Size2 =>
        dims == other.dims
      case other: Size3 =>
        other.dims._1 == dims._1 &&
        other.dims._2 == dims._2 &&
        other.dims._3 == 1
      case other: Size4 =>
        other.dims._1 == dims._1 &&
        other.dims._2 == dims._2 &&
        other.dims._3 == 1 &&
        other.dims._4 == 1
      case _ =>
        false
    })
    Size2(dims, noChannels + other.noChannels)
  }

  override def withNoTuples(noTuples: Int)
  : Size2 = {
    val height = noTuples / dims._1
    val width  = noTuples % dims._1
    Size2(width, height, noChannels)
  }

  override def withNoChannels(noChannels: Int)
  : Size2 = Size2(dims, noChannels)

  def strideY
  : Int = noChannels * dims._1

  //override def stride: (Int, Int) = (noChannels, noChannels * dims._1)

  def lineSize
  : Size1 = Size1(dims._1, noChannels)

  override protected def doToJson()
  : List[JField] = List(
    Json.field("className",  "Size2"),
    Json.field("dims_1",     dims._1),
    Json.field("dims_2",     dims._2),
    Json.field("noChannels", noChannels)
  )

}

object Size2
  extends SizeExCompanion[Size2] {

  final def apply(dims:       (Int, Int),
                  noChannels: Int)
  : Size2 = new Size2(dims, noChannels)

  final def apply(width:      Int,
                  height:     Int,
                  noChannels: Int)
  : Size2 = apply((width, height), noChannels)

  final override def derive(fields: Map[String, JValue])
  : Size2 = apply(
    Json.toInt(fields("dims_1")),
    Json.toInt(fields("dims_2")),
    Json.toInt(fields("noChannels"))
  )

  final val one
  : Size2 = apply(1, 1, 1)

  final val zero
  : Size2 = apply(0, 0, 0)

}