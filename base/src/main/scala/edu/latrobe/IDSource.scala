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

import scala.collection._
import scala.util.hashing._

final class IDSource(private var nextID:    Long,
                     val         increment: Long)
  extends Iterator[Long]
    with Serializable
    with Equatable
    with Cloneable {

  override def toString
  : String = s"IDSource[$nextID, $increment]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IDSource]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, nextID.hashCode())
    tmp = MurmurHash3.mix(tmp, increment.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: IDSource =>
      nextID    == other.nextID &&
      increment == other.increment
    case _ =>
      false
  })

  override def clone()
  : IDSource = IDSource(nextID, increment)

  override def hasNext
  : Boolean = nextID != ID.invalid

  override def next()
  : Long = {
    val id = nextID
    assume(id != ID.invalid)
    nextID += increment
    id
  }

}

object IDSource {

  final val default
  : IDSource = IDSource(1L, 1L)

  final def apply(firstID: Long)
  : IDSource = apply(firstID, 1L)

  final def apply(firstID: Long, increment: Long)
  : IDSource = new IDSource(firstID, increment)

}
