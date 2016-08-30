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

import edu.latrobe.ArrayEx
import org.bytedeco.javacpp._

final class NativeSizeT(override protected val _ptr: SizeTPointer)
  extends AutoClosingPointerEx[SizeTPointer] {
  require(_ptr != null)
}

object NativeSizeT
  extends AutoClosingPointerExCompanion[NativeSizeT, SizeTPointer, Long] {

  final override def allocate(length: Long)
  : NativeSizeT = {
    require(length > 0L)
    val ptr = new SizeTPointer(length)
    new NativeSizeT(ptr)
  }

  final override def apply(value: Long)
  : NativeSizeT = {
    val result = allocate(1L)
    result._ptr.put(value)
    result
  }

  final override def derive(array: Array[Long])
  : NativeSizeT = {
    val result = apply(array.length)
    ArrayEx.foreachPair(
      array
    )(result._ptr.put(_, _))
    result
  }

  /**
    * aka. NULL ptr
    */
  final override lazy val NULL
  : NativeSizeT = new NativeSizeT(new SizeTPointer())

}
