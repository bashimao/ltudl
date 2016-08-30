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
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.cublaze.modules._
import edu.latrobe.native.NativeFloat
import edu.latrobe.sizes.{Size1, Size2}
import org.bytedeco.javacpp.cuda._
import org.json4s.JsonAST._

import scala.collection._
import scala.util.hashing._

@SerialVersionUID(1L)
object CUBlaze
  extends Plugin {

  final def name
  : String = "CUBlazePlugin"

  final override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), name.hashCode)

  final override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Plugin]

  final override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Plugin =>
      name == other.name
    case _ =>
      false
  })

  override def platforms
  : Set[IndependentPlatform] = Set(CUDA)

  override protected def doLoad()
  : Unit = {
    // Makes sure that javacpp/CUDA dependencies are loaded before doing anything else.
    cudaPeekAtLastError()

    Platform.register(
      CUDA,
      (tensor, context) => {
        val device = context.asInstanceOf[LogicalDevice]
        tensor.toCUDARealTensor(device)
      },
      (tensor, context) => {
        val device = context.asInstanceOf[LogicalDevice]
        tensor.asOrToCUDARealTensor(device)
      }
    )

    AddBiasBuilder.register(2, AddBias_CUDA_Baseline_Description)
    BatchNormalizationBuilder.register(2, BatchNormalization_CUDA_CUDNN_Description)
    ConvolutionFilterBuilder.register(2, ConvolutionFilter_CUDA_CUDNN_Description)
    DropoutBuilder.register(2, Dropout_CUDA_CUDNN_Description)
    MultiplyFilterBuilder.register(2, ImmediateFilter_CUDA_Baseline_Description)
    LinearFilterBuilder.register(2, LinearFilter_CUDA_CUBLAS_Description)
    LogSoftmaxBuilder.register(2, LogSoftmax_CUDA_CUDNN_Description)
    MaxPoolingBuilder.register(2, MaxPooling_CUDA_CUDNN_Description)
    MeanPoolingBuilder.register(2, MeanPooling_CUDA_CUDNN_Description)
    ReLUBuilder.register(2, ReLU_CUDA_CUDNN_Description)
    SReLUBuilder.register(2, SReLU_CUDA_CUDNN_Description)
    SigmoidBuilder.register(2, Sigmoid_CUDA_CUDNN_Description)
    SoftmaxBuilder.register(2, Softmax_CUDA_CUDNN_Description)
    SwitchPlatformBuilder.register(2, SwitchPlatform_CUDA_Baseline_Description)
    TanhBuilder.register(2, Tanh_CUDA_CUDNN_Description)
  }

  override protected def doUnload()
  : Unit = {
    // Makes sure that javacpp/CUDA dependencies are loaded before doing anything else.
    cudaPeekAtLastError()

    AddBiasBuilder.unregister(AddBias_CUDA_Baseline_Description)
    BatchNormalizationBuilder.unregister(BatchNormalization_CUDA_CUDNN_Description)
    ConvolutionFilterBuilder.unregister(ConvolutionFilter_CUDA_CUDNN_Description)
    DropoutBuilder.unregister(Dropout_CUDA_CUDNN_Description)
    MultiplyFilterBuilder.unregister(ImmediateFilter_CUDA_Baseline_Description)
    LinearFilterBuilder.unregister(LinearFilter_CUDA_CUBLAS_Description)
    LogSoftmaxBuilder.unregister(LogSoftmax_CUDA_CUDNN_Description)
    MaxPoolingBuilder.unregister(MaxPooling_CUDA_CUDNN_Description)
    MeanPoolingBuilder.unregister(MeanPooling_CUDA_CUDNN_Description)
    ReLUBuilder.unregister(ReLU_CUDA_CUDNN_Description)
    SReLUBuilder.unregister(SReLU_CUDA_CUDNN_Description)
    SigmoidBuilder.unregister(Sigmoid_CUDA_CUDNN_Description)
    SoftmaxBuilder.unregister(Softmax_CUDA_CUDNN_Description)
    SwitchPlatformBuilder.unregister(SwitchPlatform_CUDA_Baseline_Description)
    TanhBuilder.unregister(Tanh_CUDA_CUDNN_Description)
  }

  override def collectRuntimeStatus()
  : JObject = {
    logger.trace("Collecting runtime status for edu.latrobe.cublaze")

    val fields = List.newBuilder[JField]

    fields += Json.field("CUDA.version", _CUDA.version)
    fields += Json.field("CUDA.driverVersion", _CUDA.driverVersion)
    fields += Json.field("CUDA.runtimeVersion", _CUDA.runtimeVersion)
    try {
      fields += Json.field("CUBLAS.version", _CUBLAS.version)
    }
    catch {
      case e: Exception =>
        fields += Json.field("CUBLAS.version", s"ERROR: $e")
    }
    try {
      fields += Json.field("NPP.version", _NPP.version.toString)
    }
    catch {
      case e: Exception =>
        fields += Json.field("NPP.version", s"ERROR: $e")
    }

    fields += Json.field("_RealDeviceBuffer.dataType",       _RealDeviceBuffer.dataType)
    fields += Json.field("_RealTensorDeviceBuffer.dataType", _RealTensorDeviceBuffer.dataType)
    try {
      fields += Json.field("CUDNN.version", _CUDNN.version)
    }
    catch {
      case e: Exception =>
        fields += Json.field("CUDNN.version", s"ERROR: $e")
    }

    fields += Json.field("devices", PhysicalDevice.collectRuntimeStatus())

    fields += Json.field("CUBLAZE_NO_LOGICAL_DEVICES", CUBLAZE_NO_LOGICAL_DEVICES)
    fields += Json.field("CUBLAZE_NO_STAGING_BUFFERS", CUBLAZE_NO_STAGING_BUFFERS)
    fields += Json.field("CUBLAZE_STAGING_BUFFER_SIZE", CUBLAZE_STAGING_BUFFER_SIZE)
    fields += Json.field("CUBLAZE_SCRATCH_BUFFER_SIZE", CUBLAZE_SCRATCH_BUFFER_SIZE)

    JObject(fields.result())
  }

}
