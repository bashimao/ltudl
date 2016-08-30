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

package edu.latrobe.blaze.sinks

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io._
import java.io._
import org.apache.commons.codec.binary._
import org.json4s.JsonAST._
import org.json4s.jackson._
import scala.util.hashing._

final class FileSink(override val builder: FileSinkBuilder,
                     override val seed:    InstanceSeed)
  extends StreamBackedSink[FileSinkBuilder] {

  val handle
  : FileHandle = builder.handle

  val useBuffer
  : Boolean = builder.useBuffer

  val stream
  : OutputStream = {
    val outStr = handle.createStream(append = true)
    val bufStr = {
      if (useBuffer) {
        new BufferedOutputStream(outStr)
      }
      else {
        outStr
      }
    }
    bufStr
  }

  override def write(src0: Any)
  : Unit = {
    val tmp = new PrintStream(stream)
    src0 match {
      case src0: String =>
        tmp.print(src0)
      case src0: Array[Byte] =>
        tmp.print(Base64.encodeBase64String(src0))
      case src0: JValue =>
        tmp.print(JsonMethods.compact(src0))
      case src0: JsonSerializable =>
        tmp.print(JsonMethods.compact(src0.toJson))
      case _ =>
        tmp.print(src0)
    }
    tmp.flush()
  }

  override def writeRaw(src0: Array[Byte])
  : Unit = stream.write(src0)

  override def writeRaw(src0: JSerializable)
  : Unit = {
    val tmp = new ObjectOutputStream(stream)
    tmp.writeObject(src0)
    tmp.flush()
  }

  override protected def doClose()
  : Unit = {
    stream.flush()
    stream.close()
    super.doClose()
  }

}

final class FileSinkBuilder
  extends StreamBackedSinkBuilder[FileSinkBuilder] {

  override def repr
  : FileSinkBuilder = this

  private var _handle
  : FileHandle = LocalFileHandle("default.out")

  def handle
  : FileHandle = _handle

  def handle_=(value: FileHandle)
  : Unit = {
    require(value != null)
    _handle = value
  }

  def setHandle(value: FileHandle)
  : FileSinkBuilder = {
    handle_=(value)
    this
  }

  var useBuffer
  : Boolean = true

  def setUseBuffer(value: Boolean)
  : FileSinkBuilder = {
    useBuffer_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _handle :: useBuffer :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _handle.hashCode())
    tmp = MurmurHash3.mix(tmp, useBuffer.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[FileSinkBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: FileSinkBuilder =>
      _handle   == other._handle &&
      useBuffer == other.useBuffer
    case _ =>
      false
  })

  override protected def doCopy()
  : FileSinkBuilder = FileSinkBuilder()


  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: FileSinkBuilder =>
        other._handle   = _handle
        other.useBuffer = useBuffer
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : FileSink = new FileSink(this, seed)

}

object FileSinkBuilder {

  final def apply()
  : FileSinkBuilder = new FileSinkBuilder

  final def apply(handle: FileHandle)
  : FileSinkBuilder = apply().setHandle(handle)

}
