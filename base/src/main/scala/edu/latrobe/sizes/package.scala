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

package object sizes {

  type TemporalSize = Size1

  type Size1D = Size1

  type SpatialSize = Size2

  type Size2D = Size2

  type VolumetricSize = Size3

  type Size3D = Size3

  type HyperCubeSize = Size4

  type Size4D = Size4

  final val TemporalSize = Size1

  final val Size1D = Size1

  final val SpatialSize = Size2

  final val Size2D = Size2

  final val VolumetricSize = Size3

  final val Size3D = Size3

  final val HyperCubeSize = Size4

  final val Size4D = Size4

  final implicit class SizeFunctions(s: Size) {

    def toSize1
    : Size1 = Size1(s.noTuples, s.noChannels)

  }

}
