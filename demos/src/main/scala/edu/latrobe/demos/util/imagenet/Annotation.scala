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
import edu.latrobe.sizes._
import org.w3c.dom._
import scala.util.hashing._
import spire.implicits._

/**
 * A image annotation.
 */
final class Annotation(val folder:      String,
                       val filename:    String,
                       val source:      String,
                       val size:        Size2,
                       val isSegmented: Boolean,
                       val bodies:      Array[Body])
  extends Serializable
    with Equatable {
  require(
    folder   != null &&
    filename != null &&
    source   != null &&
    size     != null &&
    !ArrayEx.contains(bodies, null)
  )


  override def toString
  : String = {
    s"Annotation[$folder, $filename, $source, $size, $isSegmented, ${bodies.length}]"
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Annotation]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, folder.hashCode())
    tmp = MurmurHash3.mix(tmp, filename.hashCode())
    tmp = MurmurHash3.mix(tmp, source.hashCode())
    tmp = MurmurHash3.mix(tmp, size.hashCode())
    tmp = MurmurHash3.mix(tmp, isSegmented.hashCode())
    tmp = MurmurHash3.mix(tmp, ArrayEx.hashCode(bodies))
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Annotation =>
      folder      == other.folder      &&
      filename    == other.filename    &&
      source      == other.source      &&
      size        == other.size        &&
      isSegmented == other.isSegmented &&
      ArrayEx.compare(bodies, other.bodies)
    case _ =>
      false
  })

}

object Annotation {

  final def apply(folder:      String,
                  filename:    String,
                  source:      String,
                  size:        Size2,
                  isSegmented: Boolean,
                  bodies:      Array[Body])
  : Annotation = new Annotation(
    folder, filename, source, size, isSegmented, bodies
  )

  final def derive(e: Element)
  : Annotation = {
    val folder = {
      val nodes = e.getElementsByTagName("folder")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent
      }
      else {
        throw new IndexOutOfBoundsException
      }

    }

    val filename = {
      val nodes = e.getElementsByTagName("filename")
      if (nodes.getLength == 1) {
        nodes.item(0).getTextContent
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val source = {
      val nodes = e.getElementsByTagName("source")
      if (nodes.getLength == 1) {
        val node = nodes.item(0).asInstanceOf[Element]
        val dbs = node.getElementsByTagName("database")
        if (dbs.getLength < 1) {
          "Unknown"
        }
        else if (dbs.getLength == 1) {
          s"Database: ${dbs.item(0).asInstanceOf[Element].getTextContent}"
        }
        else {
          throw new IndexOutOfBoundsException
        }
      }
      else {
        throw new IndexOutOfBoundsException
      }
    }

    val size = {
      val nodes = e.getElementsByTagName("size")
      if (nodes.getLength != 1) {
        throw new IndexOutOfBoundsException
      }
      val node = nodes.item(0).asInstanceOf[Element]
      val w = {
        val nodes = node.getElementsByTagName("width")
        if (nodes.getLength != 1) {
          throw new IndexOutOfBoundsException
        }
        nodes.item(0).getTextContent.toInt
      }
      val h = {
        val nodes = node.getElementsByTagName("height")
        if (nodes.getLength != 1) {
          throw new IndexOutOfBoundsException
        }
        nodes.item(0).getTextContent.toInt
      }
      val d = {
        val nodes = node.getElementsByTagName("depth")
        if (nodes.getLength != 1) {
          throw new IndexOutOfBoundsException
        }
        nodes.item(0).getTextContent.toInt
      }
      Size2(w, h, d)
    }

    val isSegmented = {
      val nodes = e.getElementsByTagName("segmented")
      if (nodes.getLength > 1) {
        throw new IndexOutOfBoundsException
      }
      else if (nodes.getLength > 0) {
        nodes.item(0).getTextContent == "1"
      }
      else {
        false
      }
    }

    val bodies = {
      val nodes = e.getElementsByTagName("object")
      val tmp = new Array[Body](nodes.getLength)
      cfor(0)(_ < tmp.length, _ + 1)(
        i => tmp(i) = Body.fromXml(nodes.item(i).asInstanceOf[Element])
      )
      tmp
    }

    apply(folder, filename, source, size, isSegmented, bodies)
  }

}
