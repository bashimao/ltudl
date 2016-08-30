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

import org.bytedeco.javacpp._

final class NativeShort(override protected val _ptr: ShortPointer)
  extends AutoClosingPointerEx[ShortPointer] {
  require(_ptr != null)
}

object NativeShort
  extends AutoClosingPointerExCompanion[NativeShort, ShortPointer, Short] {

  final override def allocate(length: Long)
  : NativeShort = {
    require(length > 0L)
    val ptr = new ShortPointer(length)
    new NativeShort(ptr)
  }

  final override def apply(value: Short)
  : NativeShort = {
    val result = allocate(1L)
    result._ptr.put(value)
    result
  }

  final override def derive(array: Array[Short])
  : NativeShort = {
    val result = allocate(array.length)
    result._ptr.put(array, 0, array.length)
    result
  }

  final override lazy val NULL
  : NativeShort = new NativeShort(new ShortPointer())

}
