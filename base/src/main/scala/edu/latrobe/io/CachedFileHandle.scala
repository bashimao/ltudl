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

package edu.latrobe.io

import edu.latrobe._
import edu.latrobe.native._
import java.io._
import java.nio._
import java.nio.channels._
import org.json4s.JsonAST._
import scala.util.hashing._

/**
  * A wrapper around a file handle that uses native caching.
 */
@SerialVersionUID(1L)
final class CachedFileHandle(val handle: FileHandle)
  extends FileHandle {

  override def toString
  : String = s"Cached[$handle]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), handle.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CachedFileHandle]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CachedFileHandle =>
      handle == other.handle
    case _ =>
      false
  })

  override def isDirectory
  : Boolean = handle.isDirectory

  override def ++(namePart: String)
  : CachedFileHandle = CachedFileHandle(handle ++ namePart)

  override def withoutExtension()
  : FileHandle = CachedFileHandle(handle.withoutExtension())

  // ---------------------------------------------------------------------------
  //    Read/Write
  // ---------------------------------------------------------------------------
  @transient
  private lazy val _cache
  : ByteBuffer = handle.readAsBuffer()

  override def createStream(append: Boolean)
  : OutputStream = throw new UnsupportedOperationException

  override def createChannel(append: Boolean)
  : WritableByteChannel = throw new UnsupportedOperationException

  override def openStream()
  : InputStream = ByteBufferBackedInputStream(readAsBuffer())

  override def openChannel()
  : ReadableByteChannel = Channels.newChannel(openStream())

  override def readAsArray()
  : Array[Byte] = NativeBufferEx.toArray(readAsBuffer())

  override def readAsBuffer()
  : ByteBuffer = _cache.duplicate()


  // ---------------------------------------------------------------------------
  //    Conversion / more information.
  // --------------------------------------------------------------------------
  override protected def doToJson()
  : List[JField] = List(
    Json.field("handle", handle)
  )


  // ---------------------------------------------------------------------------
  //    Directory services.
  // ---------------------------------------------------------------------------
  override protected def doTraverse(depth:       Int,
                                    onDirectory: (Int, FileHandle) => Boolean,
                                    onFile:      (Int, FileHandle) => Unit)
  : Unit = {
    require(depth == 0)
    handle.traverse(onDirectory, onFile)
  }

}

object CachedFileHandle {

  final def apply(handle: FileHandle)
  : CachedFileHandle = new CachedFileHandle(handle)

}
