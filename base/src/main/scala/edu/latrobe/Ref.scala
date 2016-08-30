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

import java.io.{ObjectInputStream, ObjectOutputStream}
import scala.util.hashing._

@SerialVersionUID(1L)
final class Ref[T <: ClosableEx](val value: T)
  extends Equatable
    with Cloneable
    with Serializable
    with AutoClosing {
  value.incrementReferenceCount()

  override def toString
  : String = s"Ref[$value]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), value.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Ref[T]]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Ref[T] =>
      value == other.value
    case _ =>
      false
  })

  override def clone()
  : Ref[T] = new Ref(value)

  override protected def doClose()
  : Unit = {
    value.decrementReferenceCount()
    super.doClose()
  }

  private def readObject(stream: ObjectInputStream)
  : Unit = {
    stream.defaultReadObject()
    value.incrementReferenceCount()
  }

  private def writeObject(stream: ObjectOutputStream)
  : Unit = {
    stream.defaultWriteObject()
  }

}

object Ref {

  final def apply[T <: ClosableEx](value: T)
  : Ref[T] = new Ref(value)

}
