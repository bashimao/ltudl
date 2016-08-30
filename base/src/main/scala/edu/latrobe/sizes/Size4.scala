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

final class Size4(override val dims:       (Int, Int, Int, Int),
                  override val noChannels: Int)
  extends SizeEx[Size4]
    with CartesianSize[Size4, (Int, Int, Int, Int)]  {
  require(
    dims._1 >= 0 &&
    dims._2 >= 0 &&
    dims._3 >= 0 &&
    dims._4 >= 0 &&
    noChannels >= 0
  )

  override def toString
  : String = s"Size4[$dims x $noChannels]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Size4]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, dims.hashCode())
    tmp = MurmurHash3.mix(tmp, noChannels.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Size4 =>
      dims       == other.dims &&
      noChannels == other.noChannels
    case _ =>
      false
  })

  override val noTuples
  : Int = dims._1 * dims._2 * dims._3 * dims._4

  override val noValues
  : Int = noTuples * noChannels

  override def multiplex
  : Size = {
    require(noChannels == 1)
    Size3(dims._1, dims._2, dims._3, dims._4)
  }

  override def demultiplex
  : Size = throw new UnsupportedOperationException

  override def ++(other: Size)
  : Size4 = {
    require(noChannels == other.noChannels)
    other match {
      case other: Size1 =>
        require(dims._1 == other.noTuples)
        require(dims._3 == 1)
        require(dims._4 == 1)
        Size4(
          dims._1,
          dims._2 + 1,
          1,
          1,
          noChannels
        )
      case other: Size2 =>
        require(dims._1 == other.dims._1)
        require(dims._2 == other.dims._2)
        require(dims._4 == 1)
        Size4(
          dims._1,
          dims._2,
          dims._3 + 1,
          1,
          noChannels
        )
      case other: Size3 =>
        require(dims._1 == other.dims._1)
        require(dims._2 == other.dims._2)
        require(dims._3 == other.dims._3)
        Size4(
          dims._1,
          dims._2,
          dims._3,
          dims._4 + 1,
          noChannels
        )
      case other: Size4 =>
        require(dims._1 == other.dims._1)
        require(dims._2 == other.dims._2)
        require(dims._3 == other.dims._3)
        Size4(
          dims._1,
          dims._2,
          dims._3,
          dims._4 + other.dims._4,
          noChannels
        )
      case _ =>
        throw new MatchError(other)
    }
  }

  override def :++(other: Size)
  : Size4 = {
    require(other match {
      case other: Size1 =>
        dims._1 == other.noTuples &&
        dims._2 == 1 &&
        dims._3 == 1 &&
        dims._4 == 1
      case other: Size2 =>
        dims._1 == other.dims._1 &&
        dims._2 == other.dims._2 &&
        dims._3 == 1 &&
        dims._4 == 1
      case other: Size3 =>
        dims._1 == other.dims._1 &&
        dims._2 == other.dims._2 &&
        dims._3 == other.dims._3 &&
        dims._4 == 1
      case other: Size4 =>
        dims == other.dims
      case _ =>
        false
    })
    Size4(dims, noChannels + other.noChannels)
  }

  override def withNoTuples(noTuples: Int)
  : Size4 = {
    val time   = noTuples / (dims._1 * dims._2 * dims._3)
    val timeR  = noTuples % (dims._1 * dims._2 * dims._3)
    val depth  = timeR / (dims._1 * dims._2)
    val depthR = timeR % (dims._1 * dims._2)
    val height = depthR / dims._1
    val width  = depthR % dims._1
    Size4(width, height, depth, time, noChannels)
  }

  override def withNoChannels(noChannels: Int)
  : Size4 = Size4(dims, noChannels)

  def strideY
  : Int = noChannels * dims._1

  def strideZ
  : Int = noChannels * dims._1 * dims._2

  def strideW
  : Int = noChannels * dims._1 * dims._2 * dims._3

  def stride
  : (Int, Int, Int, Int) = {
    val x = noChannels
    val y = x * dims._1
    val z = y * dims._2
    val w = z * dims._3
    (x, y, z, w)
  }

  def lineSize
  : Size1 = Size1(dims._1, noChannels)

  def planeSize
  : Size2 = Size2(dims._1, dims._2, noChannels)

  def boxSize
  : Size3 = Size3(dims._1, dims._2, dims._3, noChannels)

  override protected def doToJson()
  : List[JField] = List(
    Json.field("className",  "Size4"),
    Json.field("dims_1",     dims._1),
    Json.field("dims_2",     dims._2),
    Json.field("dims_3",     dims._3),
    Json.field("dims_4",     dims._4),
    Json.field("noChannels", noChannels)
  )

}

object Size4
  extends SizeExCompanion[Size4] {

  final def apply(dims:       (Int, Int, Int, Int),
                  noChannels: Int)
  : Size4 = new Size4(dims, noChannels)

  final def apply(width:      Int,
                  height:     Int,
                  depth:      Int,
                  time:       Int,
                  noChannels: Int)
  : Size4 = apply((width, height, depth, time), noChannels)

  final override def derive(fields: Map[String, JValue])
  : Size4 = apply(
    Json.toInt(fields("dims_1")),
    Json.toInt(fields("dims_2")),
    Json.toInt(fields("dims_3")),
    Json.toInt(fields("dims_4")),
    Json.toInt(fields("noChannels"))
  )

  final val one
  : Size4 = apply(1, 1, 1, 1, 1)

  final val zero
  : Size4 = apply(0, 0, 0, 0, 0)

}
