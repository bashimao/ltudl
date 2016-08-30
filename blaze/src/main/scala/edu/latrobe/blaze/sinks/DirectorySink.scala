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

/**
  * Each write will create a new file in the directory! Be careful!
  */
final class DirectorySink(override val builder: DirectorySinkBuilder,
                          override val seed:    InstanceSeed)
  extends SinkEx[DirectorySinkBuilder] {

  val handle
  : FileHandle = builder.handle

  val noFilesMax
  : Int = builder.noFilesMax

  val filenameFormatFn
  : Int => String = builder.filenameFormatFn

  val bufferStream
  : Boolean = builder.useBuffer

  val useBuffer
  : Boolean = builder.useBuffer

  private var fileNo
  : Int = 0

  private def createOutputStream()
  : OutputStream = {
    // Rewind iterator if necessary.
    if (fileNo >= noFilesMax) {
      fileNo = 0
    }

    // Assemble filename.
    val filename = filenameFormatFn(fileNo)
    val file = handle ++ filename

    // Assemble stream.
    val outStr = file.createStream(append = false)
    if (useBuffer) {
      new BufferedOutputStream(outStr)
    }
    else {
      outStr
    }
  }

  override def write(src0: Any)
  : Unit = {
    using(new PrintStream(createOutputStream()))(stream => {
      src0 match {
        case src0: String =>
          stream.print(src0)
        case src0: Array[Byte] =>
          stream.print(Base64.encodeBase64String(src0))
        case src0: JValue =>
          stream.print(JsonMethods.compact(src0))
        case src0: JsonSerializable =>
          stream.print(JsonMethods.compact(src0.toJson))
        case _ =>
          stream.print(src0)
      }
      stream.flush()
    })
  }

  override def writeRaw(src0: Array[Byte])
  : Unit = {
    using(createOutputStream())(stream => {
      stream.write(src0)
      stream.flush()
    })
  }

  override def writeRaw(src0: JSerializable)
  : Unit = {
    using(new ObjectOutputStream(createOutputStream()))(stream => {
      stream.writeObject(src0)
      stream.close()
    })
  }

  override def state
  : SinkState = DirectorySinkState(super.state, fileNo)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: DirectorySinkState =>
        fileNo = state.fileNo
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class DirectorySinkBuilder
  extends SinkExBuilder[DirectorySinkBuilder] {

  override def repr
  : DirectorySinkBuilder = this

  private var _handle
  : FileHandle = LocalFileHandle(".")

  def handle
  : FileHandle = _handle

  def handle_=(value: FileHandle)
  : Unit = {
    require(value != null)
    _handle = value
  }

  def setHandle(value: FileHandle)
  : DirectorySinkBuilder = {
    handle_=(value)
    this
  }

  private var _noFilesMax
  : Int = 10

  def noFilesMax
  : Int = _noFilesMax

  def noFilesMax_=(value: Int)
  : Unit = {
    require(value > 0)
    _noFilesMax = value
  }

  def setNoFilesMax(value: Int)
  : DirectorySinkBuilder = {
    noFilesMax_=(value)
    this
  }

  private var _filenameFormatFn
  : Int => String = "default_%05d.out".format(_)

  def filenameFormatFn
  : Int => String = _filenameFormatFn

  def filenameFormatFn_=(value: Int => String)
  : Unit = {
    require(value != null)
    _filenameFormatFn = value
  }

  def setFilenameFormatFn(value: Int => String)
  : DirectorySinkBuilder = {
    filenameFormatFn_=(value)
    this
  }

  var useBuffer
  : Boolean = true

  def setUseBuffer(value: Boolean)
  : DirectorySinkBuilder = {
    useBuffer_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _handle :: _noFilesMax :: _filenameFormatFn :: useBuffer :: super.doToString()
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _handle.hashCode())
    tmp = MurmurHash3.mix(tmp, _noFilesMax.hashCode())
    tmp = MurmurHash3.mix(tmp, _filenameFormatFn.hashCode())
    tmp = MurmurHash3.mix(tmp, useBuffer.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DirectorySinkBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DirectorySinkBuilder =>
      _handle           == other._handle           &&
      _noFilesMax       == other._noFilesMax       &&
      _filenameFormatFn == other._filenameFormatFn &&
      useBuffer         == other.useBuffer
    case _ =>
      false
  })

  override protected def doCopy()
  : DirectorySinkBuilder = DirectorySinkBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DirectorySinkBuilder =>
        other._handle           = _handle
        other._noFilesMax       = _noFilesMax
        other._filenameFormatFn = _filenameFormatFn
        other.useBuffer         = useBuffer
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : DirectorySink = new DirectorySink(this, seed)

}

object DirectorySinkBuilder {

  final def apply()
  : DirectorySinkBuilder = new DirectorySinkBuilder

  final def apply(handle: FileHandle)
  : DirectorySinkBuilder = apply().setHandle(handle)

  final def apply(handle:     FileHandle,
                  noFilesMax: Int)
  : DirectorySinkBuilder = apply(
    handle
  ).setNoFilesMax(noFilesMax)

  final def apply(handle:           FileHandle,
                  noFilesMax:       Int,
                  filenameFormatFn: Int => String)
  : DirectorySinkBuilder = apply(
    handle,
    noFilesMax
  ).setFilenameFormatFn(filenameFormatFn)

}

final case class DirectorySinkState(override val parent: InstanceState,
                                    fileNo:              Int)
  extends SinkState
