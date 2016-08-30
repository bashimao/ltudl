/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.demos.util.imagenet

import edu.latrobe.Equatable
import org.w3c.dom._

import scala.util.hashing._

final class BoundingBox(val min: (Int, Int),
                        val max: (Int, Int))
  extends Serializable
    with Equatable {
  require(min != null && max != null)

  override def toString
  : String = s"BoundingBox[$min, $max]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BoundingBox]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, min.hashCode())
    tmp = MurmurHash3.mix(tmp, max.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BoundingBox =>
      min == other.min &&
      max == other.max
    case _ =>
      false
  })

}

object BoundingBox {

  final def apply(min: (Int, Int),
                  max: (Int, Int))
  : BoundingBox = new BoundingBox(min, max)

  final def apply(minX: Int,
                  minY: Int,
                  maxX: Int,
                  maxY: Int)
  : BoundingBox = apply((minX, minY), (maxX, maxY))

  final def fromXml(e: Element)
  : BoundingBox = {
    val minX = {
      val nodes = e.getElementsByTagName("xmin")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent.toInt
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val minY = {
      val nodes = e.getElementsByTagName("ymin")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent.toInt
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val maX = {
      val nodes = e.getElementsByTagName("xmax")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent.toInt
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val maxY = {
      val nodes = e.getElementsByTagName("ymax")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent.toInt
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    apply(minX, minY, maX, maxY)
  }

}
