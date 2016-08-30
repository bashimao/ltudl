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

import edu.latrobe._
import org.w3c.dom._
import scala.util.hashing._

/**
 * This is called "object" in the files. However, Object is a reserved term in java.
 */
final class Body(val name:        String,
                 val pose:        String,
                 val isTruncated: Boolean,
                 val isDifficult: Boolean,
                 val boundingBox: BoundingBox)
  extends Serializable
    with Equatable {
  require(
    name        != null &&
    pose        != null &&
    boundingBox != null
  )

  override def toString
  : String = s"Body[$name, $pose, $isTruncated, $isDifficult, $boundingBox]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Body]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, name.hashCode())
    tmp = MurmurHash3.mix(tmp, pose.hashCode())
    tmp = MurmurHash3.mix(tmp, isTruncated.hashCode())
    tmp = MurmurHash3.mix(tmp, isDifficult.hashCode())
    tmp = MurmurHash3.mix(tmp, boundingBox.hashCode())
    tmp

  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Body =>
      name        == other.name &&
      pose        == other.pose &&
      isTruncated == other.isTruncated &&
      isDifficult == other.isDifficult &&
      boundingBox == other.boundingBox
    case _ =>
      false
  })

}

object Body {

  final def apply(name:        String,
                  pose:        String,
                  isTruncated: Boolean,
                  isDifficult: Boolean,
                  boundingBox: BoundingBox)
  : Body = new Body(name, pose, isTruncated, isDifficult, boundingBox)

  final def fromXml(e: Element)
  : Body = {
    val name = {
      val nodes = e.getElementsByTagName("name")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val pose = {
      val nodes = e.getElementsByTagName("pose")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val isTruncated = {
      val nodes = e.getElementsByTagName("truncated")
      if (nodes.getLength < 1) {
        false
      }
      else if (nodes.getLength == 1) {
        nodes.item(0).getTextContent == "1"
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val isDifficult = {
      val nodes = e.getElementsByTagName("difficult")
      if (nodes.getLength < 1) {
        false
      }
      else if (nodes.getLength == 1) {
        nodes.item(0).getTextContent == "1"
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val boundingBox = {
      val nodes = e.getElementsByTagName("bndbox")
      if (nodes.getLength == 1) {
        BoundingBox.fromXml(nodes.item(0).asInstanceOf[Element])
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    apply(name, pose, isTruncated, isDifficult, boundingBox)
  }

}
