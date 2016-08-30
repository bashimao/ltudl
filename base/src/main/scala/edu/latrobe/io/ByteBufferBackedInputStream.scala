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

import java.io._
import java.nio._

// More or less a straight forward Scala adaptation of what I found at stack overflow:
// http://stackoverflow.com/questions/4332264/wrapping-a-bytebuffer-with-an-inputstream/6603018#6603018
final class ByteBufferBackedInputStream(val buffer: ByteBuffer)
  extends InputStream {

  override def read(): Int = {
    if (!buffer.hasRemaining) {
      return -1
    }
    buffer.get() & 0xFF
  }

  override def read(b: Array[Byte], off: Int, len: Int)
  : Int = {
    // super.read(b, off, len)
    val remaining = buffer.remaining()
    if (remaining <= 0) {
      return -1
    }

    val n = Math.min(len, buffer.remaining)
    buffer.get(b, off, n)
    n
  }

}

object ByteBufferBackedInputStream {

  final def apply(buffer: ByteBuffer)
  : ByteBufferBackedInputStream = new ByteBufferBackedInputStream(buffer)

}