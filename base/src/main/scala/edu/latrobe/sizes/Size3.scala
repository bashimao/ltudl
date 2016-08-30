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

final class Size3(override val dims:       (Int, Int, Int),
                  override val noChannels: Int)
  extends SizeEx[Size3]
    with CartesianSize[Size3, (Int, Int, Int)] {
  require(
    dims._1 >= 0 &&
    dims._2 >= 0 &&
    dims._3 >= 0 &&
    noChannels >= 0
  )

  override def toString
  : String = s"Size3[$dims x $noChannels]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Size3]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, dims.hashCode())
    tmp = MurmurHash3.mix(tmp, noChannels.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Size3 =>
      dims       == other.dims &&
      noChannels == other.noChannels
    case _ =>
      false
  })

  override val noTuples
  : Int = dims._1 * dims._2 * dims._3

  override val noValues
  : Int = noTuples * noChannels

  override def multiplex
  : Size = {
    require(noChannels == 1)
    Size2(dims._1, dims._2, dims._3)
  }

  override def demultiplex
  : Size = Size4(dims._1, dims._2, dims._3, noChannels, 1)

  override def ++(other: Size): Size3 = {
    require(noChannels == other.noChannels)
    other match {
      case other: Size1 =>
        require(dims._1 == other.noTuples)
        require(dims._3 == 1)
        Size3(
          dims._1,
          dims._2 + 1,
          1,
          noChannels
        )
      case other: Size2 =>
        require(dims._1 == other.dims._1)
        require(dims._2 == other.dims._2)
        Size3(
          dims._1,
          dims._2,
          dims._3 + 1,
          noChannels
        )
      case other: Size3 =>
        require(dims._1 == other.dims._1)
        require(dims._2 == other.dims._2)
        Size3(
          dims._1,
          dims._2,
          dims._3 + other.dims._3,
          noChannels
        )
      case other: Size4 =>
        require(dims._1 == other.dims._1)
        require(dims._2 == other.dims._2)
        Size3(
          dims._1,
          dims._2,
          dims._3 + (other.dims._3 * other.dims._4),
          noChannels
        )
      case _ =>
        throw new MatchError(other)
    }
  }

  override def :++(other: Size): Size3 = {
    require(other match {
      case other: Size1 =>
        dims._1 == other.noTuples &&
        dims._2 == 1 &&
        dims._3 == 1
      case other: Size2 =>
        dims._1 == other.dims._1 &&
        dims._2 == other.dims._2 &&
        dims._3 == 1
      case other: Size3 =>
        dims == other.dims
      case other: Size4 =>
        other.dims._1 == dims._1 &&
        other.dims._2 == dims._2 &&
        other.dims._3 == dims._3 &&
        other.dims._4 == 1
      case _ =>
        false
    })
    Size3(dims, noChannels + other.noChannels)
  }

  override def withNoTuples(noTuples: Int)
  : Size3 = {
    val depth  = noTuples / (dims._1 * dims._2)
    val depthR = noTuples % (dims._1 * dims._2)
    val height = depthR / dims._1
    val width  = depthR % dims._1
    Size3(width, height, depth, noChannels)
  }

  override def withNoChannels(noChannels: Int)
  : Size3 = Size3(dims, noChannels)

  def strideY
  : Int = noChannels * dims._1

  def strideZ
  : Int = noChannels * dims._1 * dims._2

  def stride
  : (Int, Int, Int) = {
    val x = noChannels
    val y = x * dims._1
    val z = y * dims._2
    (x, y, z)
  }

  def lineSize
  : Size1 = Size1(dims._1, noChannels)

  def planeSize
  : Size2 = Size2(dims._1, dims._2, noChannels)

  override protected def doToJson()
  : List[JField] = List(
    Json.field("className",  "Size3"),
    Json.field("dims_1",     dims._1),
    Json.field("dims_2",     dims._2),
    Json.field("dims_3",     dims._3),
    Json.field("noChannels", noChannels)
  )

}

object Size3
  extends SizeExCompanion[Size3] {

  final def apply(dims:       (Int, Int, Int),
                  noChannels: Int)
  : Size3 = new Size3(dims, noChannels)

  final def apply(width:      Int,
                  height:     Int,
                  depth:      Int,
                  noChannels: Int)
  : Size3 = apply((width, height, depth), noChannels)

  final override def derive(fields: Map[String, JValue])
  : Size3 = apply(
    Json.toInt(fields("dims_1")),
    Json.toInt(fields("dims_2")),
    Json.toInt(fields("dims_3")),
    Json.toInt(fields("noChannels"))
  )

  final val one
  : Size3 = apply(1, 1, 1, 1)

  final val zero
  : Size3 = apply(0, 0, 0, 0)

}
