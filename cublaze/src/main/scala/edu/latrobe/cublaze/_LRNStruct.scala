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
import org.bytedeco.javacpp.cudnn._

final class _LRNStruct private
  extends AutoClosingPointerEx[cudnnLRNStruct] {

  override protected val _ptr
  : cudnnLRNStruct = _CUDNN.createLRNDescriptor()

  override protected def doClose()
  : Unit = {
    _CUDNN.destroyLRNDescriptor(this)
    super.doClose()
  }

}

private[cublaze] object _LRNStruct {

  final def apply()
  : _LRNStruct = new _LRNStruct

}