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

package edu.latrobe.demos.imagenet

import edu.latrobe.Real
import edu.latrobe.kernels._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._

object VGG2014 {

  val updIntMeanStdDev = 100L

  val updIntHistogram = 100L

  /**
   * 11 weight layers
   */
  def createFeatureExtractorA(): SequenceBuilder = {
    val kernelConv3 = Kernel2.centered((3, 3))
    val kernelPool2 = Kernel2((2, 2), (2, 2))

    val conv64  = ConvolutionFilterBuilder(kernelConv3,  64)
    val conv128 = ConvolutionFilterBuilder(kernelConv3, 128)
    val conv256 = ConvolutionFilterBuilder(kernelConv3, 256)
    val conv512 = ConvolutionFilterBuilder(kernelConv3, 512)
    val bias = AddBiasBuilder()
    val relu = ReLUBuilder()
    val pool = MaxPoolingBuilder(kernelPool2)

    SequenceBuilder(
      Array(
        // Feature Extraction 1
        conv64, bias,
        relu,
        pool,

        // Feature Extraction 2
        conv128, bias,
        relu,
        pool,

        // Feature Extraction 3
        conv256, bias,
        relu,
        conv256, bias,
        relu,
        pool,

        // Feature Extraction 4
        conv512, bias,
        relu,
        conv512, bias,
        relu,
        pool,

        // Feature Extraction 5
        conv512, bias,
        relu,
        conv512, bias,
        relu,
        pool
      )
    )
  }

  /**
   * 11 weight layers with LRN after first layer
   */
  def createFeatureExtractorALRN(): SequenceBuilder = {
    val kernelConv3 = Kernel2.centered((3, 3))
    val kernelPool2 = Kernel2((2, 2), (2, 2))

    val conv64 = ConvolutionFilterBuilder(kernelConv3, 64)
    val conv128 = ConvolutionFilterBuilder(kernelConv3, 128)
    val conv256 = ConvolutionFilterBuilder(kernelConv3, 256)
    val conv512 = ConvolutionFilterBuilder(kernelConv3, 512)
    val bias = AddBiasBuilder()
    val relu = ReLUBuilder()
    val lrn = LateralResponseNormalizationBuilder(
      n = 5, k = 2.0f, alpha = 1e-4f, beta = 0.75f
    )
    val pool = MaxPoolingBuilder(kernelPool2)

    SequenceBuilder(
      // Feature Extraction 1
      conv64, bias,
      relu,
      lrn,
      pool,

      // Feature Extraction 2
      conv128, bias,
      relu,
      pool,

      // Feature Extraction 3
      conv256, bias,
      relu,
      conv256, bias,
      relu,
      pool,

      // Feature Extraction 4
      conv512, bias,
      relu,
      conv512, bias,
      relu,
      pool,

      // Feature Extraction 5
      conv512, bias,
      relu,
      conv512, bias,
      relu,
      pool
    )
  }

  /**
   * 13 weight layers
   */
  def createFeatureExtractorB(): SequenceBuilder = {
    val kernelConv3 = Kernel2.centered((3, 3))
    val kernelPool2 = Kernel2((2, 2), (2, 2))

    val conv64 = ConvolutionFilterBuilder(kernelConv3, 64)
    val conv128 = ConvolutionFilterBuilder(kernelConv3, 128)
    val conv256 = ConvolutionFilterBuilder(kernelConv3, 256)
    val conv512 = ConvolutionFilterBuilder(kernelConv3, 512)
    val bias = AddBiasBuilder()
    val relu = ReLUBuilder()
    val pool = MaxPoolingBuilder(kernelPool2)

    SequenceBuilder(
      // Feature Extraction 1
      conv64, bias,
      relu,
      conv64, bias,
      relu,
      pool,

      // Feature Extraction 2
      conv128, bias,
      relu,
      conv128, bias,
      relu,
      pool,

      // Feature Extraction 3
      conv256, bias,
      relu,
      conv256, bias,
      relu,
      pool,

      // Feature Extraction 4
      conv512, bias,
      relu,
      conv512, bias,
      relu,
      pool,

      // Feature Extraction 5
      conv512, bias,
      relu,
      conv512, bias,
      relu,
      pool
    )
  }

  /**
   * 16 weight layers
   */
  def createFeatureExtractorC(): SequenceBuilder = {
    val kernelConv3 = Kernel2.centered((3, 3))
    val kernelConv1 = Kernel2.centered((1, 1))
    val kernelPool2 = Kernel2((2, 2), (2, 2))

    val conv_3_64 = ConvolutionFilterBuilder(kernelConv3, 64)
    val conv_3_128 = ConvolutionFilterBuilder(kernelConv3, 128)
    val conv_3_256 = ConvolutionFilterBuilder(kernelConv3, 256)
    val conv_3_512 = ConvolutionFilterBuilder(kernelConv3, 512)
    val conv_1_256 = ConvolutionFilterBuilder(kernelConv1, 256)
    val conv_1_512 = ConvolutionFilterBuilder(kernelConv1, 512)
    val bias = AddBiasBuilder()
    val relu = ReLUBuilder()
    val pool = MaxPoolingBuilder(kernelPool2)
    SequenceBuilder(
      // Feature Extraction 1
      conv_3_64, bias,
      relu,
      conv_3_64, bias,
      relu,
      pool,

      // Feature Extraction 2
      conv_3_128, bias,
      relu,
      conv_3_128, bias,
      relu,
      pool,

      // Feature Extraction 3
      conv_3_256, bias,
      relu,
      conv_3_256, bias,
      relu,
      conv_1_256, bias,
      relu,
      pool,

      // Feature Extraction 4
      conv_3_512, bias,
      relu,
      conv_3_512, bias,
      relu,
      conv_1_512, bias,
      relu,
      pool,

      // Feature Extraction 5
      conv_3_512, bias,
      relu,
      conv_3_512, bias,
      relu,
      conv_1_512, bias,
      relu,
      pool
    )
  }

  /**
   * 16 weight layers
   */
  def createFeatureExtractorD(): SequenceBuilder = {
    val kernelConv3 = Kernel2.centered((3, 3))
    val kernelPool2 = Kernel2((2, 2), (2, 2))

    val conv_64 = ConvolutionFilterBuilder(kernelConv3, 64)
    val conv_128 = ConvolutionFilterBuilder(kernelConv3, 128)
    val conv_256 = ConvolutionFilterBuilder(kernelConv3, 256)
    val conv_512 = ConvolutionFilterBuilder(kernelConv3, 512)
    val bias = AddBiasBuilder()
    val relu = ReLUBuilder()
    val pool = MaxPoolingBuilder(kernelPool2)

    SequenceBuilder(
      // Feature Extraction 1
      conv_64, bias,
      relu,
      conv_64, bias,
      relu,
      pool,

      // Feature Extraction 2
      conv_128, bias,
      relu,
      conv_128, bias,
      relu,
      pool,

      // Feature Extraction 3
      conv_256, bias,
      relu,
      conv_256, bias,
      relu,
      conv_256, bias,
      relu,
      pool,

      // Feature Extraction 4
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      pool,

      // Feature Extraction 5
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      pool
    )
  }

  /**
   * 19 weight layers
   */
  def createFeatureExtractorE(): SequenceBuilder = {
    val kernelConv3 = Kernel2.centered((3, 3))
    val kernelPool2 = Kernel2((2, 2), (2, 2))

    val bias = AddBiasBuilder()
    val relu = ReLUBuilder()
    val pool = MaxPoolingBuilder(kernelPool2)
    val conv_64 = ConvolutionFilterBuilder(kernelConv3, 64)
    val conv_128 = ConvolutionFilterBuilder(kernelConv3, 128)
    val conv_256 = ConvolutionFilterBuilder(kernelConv3, 256)
    val conv_512 = ConvolutionFilterBuilder(kernelConv3, 512)

    SequenceBuilder(
      // Feature Extraction 1
      conv_64, bias,
      relu,
      conv_64, bias,
      relu,
      pool,

      // Feature Extraction 2
      conv_128, bias,
      relu,
      conv_128, bias,
      relu,
      pool,

      // Feature Extraction 3
      conv_256, bias,
      relu,
      conv_256, bias,
      relu,
      conv_256, bias,
      relu,
      conv_256, bias,
      relu,
      pool,

      // Feature Extraction 4
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      pool,

      // Feature Extraction 5
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      conv_512, bias,
      relu,
      pool
    )
  }

  def createClassifier(noClasses: Int, reluThreshold: Real)
  : SequenceBuilder = {
    val dropout = DropoutBuilder(Real.pointFive)
    val relu: ModuleBuilder = {
      if (reluThreshold == Real.zero) {
        ReLUBuilder()
      }
      else {
        SReLUBuilder(reluThreshold)
      }
    }
    val bias = AddBiasBuilder()
    val fc4096 = LinearFilterBuilder(4096)
    val fc1000 = LinearFilterBuilder(noClasses)

    // Remark: The inout should be Size2(7, 7, 512)
    SequenceBuilder(
      ReshapeBuilder(size => Size1(1, size.noValues)),
      fc4096, bias,
      relu,
      dropout,

      fc4096, bias,
      relu,
      dropout,

      fc1000, bias,
      LogSoftmaxBuilder(),
      ClassNLLConstraintBuilder()
    )
  }

  def createClassifierBN(noClasses: Int, reluThreshold: Real)
  : SequenceBuilder = {
    val bn      = BatchNormalizationBuilder()
    val dropout = DropoutBuilder(Real.pointFive)
    val relu: ModuleBuilder = {
      if (reluThreshold == Real.zero) {
        ReLUBuilder()
      }
      else {
        SReLUBuilder(reluThreshold)
      }
    }
    val bias = AddBiasBuilder()
    val fc4096 = LinearFilterBuilder(4096)
    val fc1000 = LinearFilterBuilder(noClasses)

    // Remark: The inout should be Size2(7, 7, 512)
    SequenceBuilder(
      ReshapeBuilder(size => Size1(1, size.noValues)),
      fc4096, bn,
      relu,
      dropout,

      fc4096, bn,
      relu,
      dropout,

      fc1000, bias,
      LogSoftmaxBuilder(),
      ClassNLLConstraintBuilder()
    )
  }

}
