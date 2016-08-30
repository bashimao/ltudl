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

/**
  * bankNo    >= 0
  * segmentNo =  0 -> Automatically assign a segment number.
  *           >  1 -> Use fixed segment number. (use this to link weights)
  */
abstract class BufferReference
  extends Equatable
    with Serializable
    with JsonSerializable {

  def bankNo
  : Int

  def segmentNo
  : Int

  def derive(segmentNo: Int)
  : BufferReference

}

abstract class BufferReferenceEx[T <: BufferReferenceEx[_]]
  extends BufferReference {

  override def derive(segmentNo: Int)
  : T

}
