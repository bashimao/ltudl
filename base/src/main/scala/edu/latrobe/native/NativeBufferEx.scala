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

package edu.latrobe.native

import java.nio._
import edu.latrobe._

object NativeBufferEx {

  @inline
  final def copy(buffer0: ByteBuffer)
  : ByteBuffer = {
    var tries = 0
    do {
      try {
        val result = {
          if (buffer0.isDirect) {
            ByteBuffer.allocateDirect(buffer0.capacity)
          }
          else {
            ByteBuffer.allocate(buffer0.capacity)
          }
        }
        result.put(buffer0)
        return result
      }
      catch {
        case e: OutOfMemoryError =>
          logger.error(s"Exception caught: ", e)
          System.gc()
          System.runFinalization()
      }
      tries += 1
    } while (tries < 100)
    throw new OutOfMemoryError("Unable to recover from out of memory situation!")
  }

  @inline
  final def toArray(buffer0: ByteBuffer)
  : Array[Byte] = {
    val result = new Array[Byte](buffer0.remaining)
    buffer0.get(result)
    result
  }

  @inline
  def toArray(buffer0: NativeRealBuffer)
  : Array[Real] = {
    val result = Array.ofDim[Real](buffer0.remaining)
    buffer0.get(result)
    result
  }


  // ---------------------------------------------------------------------------
  //    REAL SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  @inline
  final def asRealBuffer(buffer: ByteBuffer)
  : NativeRealBuffer = buffer.asDoubleBuffer()
  */
  // ---------------------------------------------------------------------------
  //    REAL SWITCH FLOAT
  // ---------------------------------------------------------------------------
  ///*
  @inline
  final def asRealBuffer(buffer0: ByteBuffer)
  : NativeRealBuffer = buffer0.asFloatBuffer()
  //*/
  // ---------------------------------------------------------------------------
  //    REAL SWITCH END
  // ---------------------------------------------------------------------------

}
