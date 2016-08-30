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

import edu.latrobe._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.cudnn._

final class _DoubleDeviceBuffer private(override val device: PhysicalDevice,
                                        override val _ptr:   DoublePointer)
  extends _DeviceBufferEx[_DoubleDeviceBuffer, DoublePointer] {
  require(device != null && _ptr != null)

  override def capacityInBytes
  : Long = _ptr.capacity() * DoubleEx.size

  override def allocateSibling()
  : _DoubleDeviceBuffer = _DoubleDeviceBuffer(device, _ptr.capacity())

  override def getSlicePtr(offset: Long, length: Long)
  : DoublePointer = {
    require(offset >= 0L && length >= 0L && offset + length <= _ptr.capacity())
    new DoublePointer {
      address = _ptr.address() + offset * DoubleEx.size
      capacity(length)
    }
  }

}

private[cublaze] object _DoubleDeviceBuffer
  extends DeviceBufferExCompanion[_DoubleDeviceBuffer, DoublePointer, Double] {

  final override def apply(device: PhysicalDevice, capacity: Long)
  : _DoubleDeviceBuffer = {
    val ptr = new DoublePointer()
    _CUDA.malloc(ptr, capacity * DoubleEx.size)
    ptr.capacity(capacity)
    new _DoubleDeviceBuffer(device, ptr)
  }

  final override val dataType
  : Int = CUDNN_DATA_DOUBLE

}
