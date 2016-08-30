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
import scala.collection.generic._
import scala.reflect._

final class RingBuffer[T](val buffer: Array[T])
  extends mutable.IndexedSeq[T]
    with Growable[T]
    with Serializable {

  private var offset: Int = 0

  private var used: Int = 0

  override def toString
  : String = s"RingBuffer[$offset, $used / ${buffer.length}]"

  override def length: Int = used

  override def apply(idx: Int)
  : T = {
    require(idx >= 0 && idx < used)
    buffer((offset + idx) % buffer.length)
  }

  override def update(idx: Int, elem: T)
  : Unit = {
    require(idx >= 0 && idx < used)
    buffer((offset + idx) % buffer.length) = elem
  }

  override def clear()
  : Unit = used = 0

  override def +=(elem: T)
  : this.type = {
    buffer((offset + used) % buffer.length) = elem
    if (used < buffer.length) {
      used += 1
    }
    else {
      offset = (offset + 1) % buffer.length
    }
    this
  }

  def isFull: Boolean = used == buffer.length

}

object RingBuffer {

  final def apply[T](capacity: Int)
                    (implicit tagT: ClassTag[T])
  : RingBuffer[T] = new RingBuffer[T](new Array[T](capacity))

}
