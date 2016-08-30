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

import org.bytedeco.javacpp._

package object native {

  type HalfPointer = ShortPointer


  // ---------------------------------------------------------------------------
  //    REAL SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  type RealPointer = DoublePointer

  type NativeReal = NativeDouble

  final val NativeReal = NativeDouble
  */
  // ---------------------------------------------------------------------------
  //    REAL SWITCH FLOAT
  // ---------------------------------------------------------------------------
  ///*
  type RealPointer = FloatPointer

  type NativeReal = NativeFloat

  final val NativeReal = NativeFloat
  //*/
  // ---------------------------------------------------------------------------
  //    REAL SWITCH END
  // ---------------------------------------------------------------------------

  final implicit class DoublePointerFunctions(dp: DoublePointer) {

    def withOffset(offset: Long)
    : DoublePointer = {
      require(offset >= 0L && offset <= dp.capacity())
      new DoublePointer {
        address = dp.address() + offset * DoubleEx.size
        capacity(dp.capacity() - offset)
      }
    }

  }

  final implicit class FloatPointerFunctions(fp: FloatPointer) {

    def withOffset(offset: Long)
    : FloatPointer = {
      require(offset >= 0L && offset <= fp.capacity())
      new FloatPointer {
        address = fp.address() + offset * FloatEx.size
        capacity(fp.capacity() - offset)
      }
    }

  }

  final implicit class HalfPointerFunctions(fp: HalfPointer) {

    def withOffset(offset: Long)
    : HalfPointer = {
      require(offset >= 0L && offset <= fp.capacity())
      new HalfPointer {
        address = fp.address() + offset * Half.size
        capacity(fp.capacity() - offset)
      }
    }

  }

}
