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
final class ByteBufferBackedOutputStream(val buffer: ByteBuffer)
  extends OutputStream {

  override def write(b: Int)
  : Unit = buffer.put(b.toByte)

  override def write(b: Array[Byte], off: Int, len: Int)
  : Unit = {
    // super.write(b, off, len)
    buffer.put(b, off, len)
  }

}

object ByteBufferBackedOutputStream {

  final def apply(buffer: ByteBuffer)
  : ByteBufferBackedOutputStream = new ByteBufferBackedOutputStream(buffer)

}
