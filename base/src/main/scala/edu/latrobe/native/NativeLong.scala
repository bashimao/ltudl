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

final class NativeLong(override protected val _ptr: LongPointer)
  extends AutoClosingPointerEx[LongPointer] {
  require(_ptr != null)
}

object NativeLong
  extends AutoClosingPointerExCompanion[NativeLong, LongPointer, Long] {

  final override def allocate(length: Long)
  : NativeLong = {
    require(length > 0L)
    val ptr = new LongPointer(length)
    new NativeLong(ptr)
  }

  final override def apply(value: Long)
  : NativeLong = {
    val result = allocate(1L)
    result._ptr.put(value)
    result
  }

  final override def derive(array: Array[Long])
  : NativeLong = {
    val result = apply(array.length)
    result._ptr.put(array, 0, array.length)
    result
  }

  /**
    * aka. NULL ptr
    */
  final override lazy val NULL
  : NativeLong = new NativeLong(new LongPointer())

}
