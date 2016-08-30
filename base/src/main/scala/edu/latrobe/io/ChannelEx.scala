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
import java.nio._
import java.nio.channels._

object ChannelEx {

  // Works even if the sizeHint is wrong.
  @inline
  final def read(channel: ReadableByteChannel, sizeHint: Int = 8192)
  : ByteBuffer = {
    var buffer = ByteBuffer.allocateDirect(sizeHint)
    var n      = -1

    while (true) {
      // Try to read all bytes.
      n = channel.read(buffer)
      while (n > 0) {
        n = channel.read(buffer)
      }

      // If EOF reached.
      if (n < 0) {
        buffer.limit(buffer.position())
        return buffer
      }

      // Try to read one more byte to see whether we have just reached EOF.
      n = channel.read(buffer)
      if (n < 0) {
        buffer.limit(buffer.position())
        return buffer
      }

      // Well.. that worked, reallocate buffer before we try again.
      val newSizeHint = Math.max(8192L, buffer.limit() * 2L)
      if (newSizeHint > ArrayEx.maxSize) {
        throw new OutOfMemoryError("Larger than maximum array size.")
      }
      val newBuffer = ByteBuffer.allocateDirect(newSizeHint.toInt)
      newBuffer.put(buffer)
      buffer = newBuffer

      // Add the byte to the buffer.
      buffer.put(n.toByte)
    }

    throw new UnknownError
  }

}
