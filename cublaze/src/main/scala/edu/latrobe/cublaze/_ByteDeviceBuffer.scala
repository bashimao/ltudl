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

import org.bytedeco.javacpp._

final class _ByteDeviceBuffer private(override val device: PhysicalDevice,
                                      override val _ptr:   BytePointer)
  extends _DeviceBuffer[BytePointer] {
  require(device != null && _ptr != null)

  override def capacityInBytes
  : Long = _ptr.capacity()

  override def getSlicePtr(offset: Long, length: Long)
  : BytePointer = {
    require(offset >= 0L && length >= 0L && offset + length <= _ptr.capacity())
    new BytePointer {
      address = _ptr.address() + offset
      capacity(length)
    }
  }

}

private[cublaze] object _ByteDeviceBuffer
  extends DeviceBufferCompanion[_ByteDeviceBuffer, BytePointer, Byte] {

  final override def apply(device: PhysicalDevice, capacity: Long)
  : _ByteDeviceBuffer = {
    val ptr = new BytePointer()
    _CUDA.malloc(ptr, capacity)
    ptr.capacity(capacity)
    new _ByteDeviceBuffer(device, ptr)
  }

}
