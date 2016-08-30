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

final class NativeFloat(override protected val _ptr: FloatPointer)
  extends AutoClosingPointerEx[FloatPointer] {
  require(_ptr != null)
}

object NativeFloat
  extends AutoClosingPointerExCompanion[NativeFloat, FloatPointer, Float] {

  final override def allocate(length: Long)
  : NativeFloat = {
    require(length > 0L)
    val ptr = new FloatPointer(length)
    new NativeFloat(ptr)
  }

  final override def apply(value: Float)
  : NativeFloat = {
    val result = allocate(1L)
    result._ptr.put(value)
    result
  }

  final override def derive(array: Array[Float])
  : NativeFloat = {
    val result = allocate(array.length)
    result._ptr.put(array, 0, array.length)
    result
  }

  /**
    * A constant point to -1.0 value. Be careful to not change the value of this.
    */
  final lazy val minusOne
  : NativeFloat = apply(-1.0f)

  /**
    * aka. NULL ptr
    */
  final override lazy val NULL
  : NativeFloat = new NativeFloat(new FloatPointer())

  /**
    * A constant point to 1.0 value. Be careful to not change the value of this.
    */
  final lazy val one
  : NativeFloat = apply(1.0f)

  /**
    * A constant point to 0.5 value. Be careful to not change the value of this.
    */
  final lazy val pointFive
  : NativeFloat = apply(0.5f)

  /**
    * A constant point to 0.0 value. Be careful to not change the value of this.
    */
  final lazy val zero
  : NativeFloat = apply(0.0f)

}