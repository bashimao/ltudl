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

package edu.latrobe.cublaze

import edu.latrobe.native._
import org.bytedeco.javacpp.nppc._

final class _SizeStruct private(val width: Int,
                                val height: Int,
                                val pixelSize: Int)
  extends AutoClosingPointerEx[NppiSize] {

  override protected val _ptr
  : NppiSize = new NppiSize()
  _ptr.width(width)
  _ptr.height(height)


  val linePitch
  : Int = width * pixelSize

  override protected def doClose()
  : Unit = {
    _ptr.deallocate()
    super.doClose()
  }

}


object _SizeStruct {

  final def apply(width: Int, height: Int, pixelSize: Int)
  : _SizeStruct = new _SizeStruct(width, height, pixelSize)

}
