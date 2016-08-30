/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe

package object kernels {

  type TemporalKernel = Kernel1

  type Kernel1D = Kernel1

  type SpatialKernel = Kernel2

  type Kernel2D = Kernel2

  type VolumetricKernel = Kernel3

  type Kernel3D = Kernel3

  type HyperCubeKernel = Kernel4

  type Kernel4D = Kernel4

  final val TemporalKernel = Kernel1

  final val Kernel1D = Kernel1

  final val SpatialKernel = Kernel2

  final val Kernel2D = Kernel2

  final val VolumetricKernel = Kernel3

  final val Kernel3D = Kernel3

  final val HyperCubeKernel = Kernel4

  final val Kernel4D = Kernel4

}
