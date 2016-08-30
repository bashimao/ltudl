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

final class PlatformTable(private val platforms: Array[Platform])
  extends DependentPlatform
    with TableLike[Platform] {
  require(!ArrayEx.contains(platforms,  null))

  override def toString
  : String = s"Table[${platforms.length}]"

  override def hashCode()
  : Int = ArrayEx.hashCode(platforms)

  override def equals(obj: Any): Boolean = obj match {
    case obj: PlatformTable =>
      ArrayEx.compare(
        platforms,
        obj.platforms
      )
    case _ =>
      false
  }

  override def length
  : Int = platforms.length

  override def getEntry(index: Int)
  : Platform = {
    val arrayIndex = {
      if (index >= 0) {
        index
      }
      else {
        platforms.length + index
      }
    }
    require(arrayIndex >= 0 && arrayIndex < platforms.length)
    platforms(arrayIndex)
  }

  override def iterator
  : Iterator[Platform] = platforms.iterator

  override protected def doToJson()
  : List[JField] = List(
    Json.field("platforms", platforms)
  )


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override def toEdgeLabel
  : String = {
    val builder = StringBuilder.newBuilder
    ArrayEx.foreach(
      platforms
    )(platform => builder ++= platform.toEdgeLabel ++= ", ")
    if (builder.nonEmpty) {
      builder.length = builder.length - 2
    }
    s"[${builder.result()}]"
  }

}

object PlatformTable {

  final def apply(platforms: Array[Platform])
  : PlatformTable = new PlatformTable(platforms)

  final def derive(platform0: Platform)
  : PlatformTable = apply(Array(platform0))

  final def derive(platform0: Platform,
                   platforms: Platform*)
  : PlatformTable = apply((platform0 :: platforms.toList).toArray)

  final def derive(platforms: Seq[Platform])
  : PlatformTable = apply(platforms.toArray)

}