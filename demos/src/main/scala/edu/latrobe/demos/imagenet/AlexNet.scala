/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (m.langer@latrobe.edu.au)
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


import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.initializers._
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.regularizers._
import edu.latrobe.kernels._
import edu.latrobe.sizes._
import scala.collection._

/**
 * Based on original paper and:
 * https://github.com/facebook/fbcunn/blob/master/examples/imagenet/models/alexnet_cudnn.lua
 */
object AlexNet {

  final def createUnifiedFeatureExtractor()
  : ModuleBuilder = {
    // Kernels
    val kernelConv11 = Kernel2((11, 11), (4, 4), (2, 2))
    val kernelConv05 = Kernel2.centered((5, 5))
    val kernelConv03 = Kernel2.centered((3, 3))
    val kernelPool   = Kernel2((3, 3), (2, 2))

    // Unique layers
    val relu = ReLUBuilder()
    val lrn  = LateralResponseNormalizationBuilder(n = 5, k = 2, alpha = 1e-4f, beta = 0.75f)
    val pool = MaxPoolingBuilder(kernelPool)

    // Assemble feature extractor.
    val features = SequenceBuilder(
      ConvolutionFilterBuilder(kernelConv11,  96).permuteWeightReferences(_.derive(100)),
      AddBiasBuilder().permuteWeightReferences(_.derive(101)),
      relu,
      pool, /* lrn,*/

      ConvolutionFilterBuilder(kernelConv05, 256).permuteWeightReferences(_.derive(110)),
      AddBiasBuilder().permuteWeightReferences(_.derive(111)),
      relu,
      pool, /* lrn,*/

      ConvolutionFilterBuilder(kernelConv03, 384).permuteWeightReferences(_.derive(120)),
      AddBiasBuilder().permuteWeightReferences(_.derive(121)),
      relu,

      ConvolutionFilterBuilder(kernelConv03, 384).permuteWeightReferences(_.derive(130)),
      AddBiasBuilder().permuteWeightReferences(_.derive(131)),
      relu,

      ConvolutionFilterBuilder(kernelConv03, 256).permuteWeightReferences(_.derive(140)),
      AddBiasBuilder().permuteWeightReferences(_.derive(141)),
      relu,
      pool
    )
    features
  }

  final def createSingleColumnFeatureExtractor()
  : ModuleBuilder = {
    // Kernels
    val kernelConv11 = Kernel2((11, 11), (4, 4), (2, 2))
    val kernelConv05 = Kernel2.centered((5, 5))
    val kernelConv03 = Kernel2.centered((3, 3))
    val kernelPool   = Kernel2((3, 3), (2, 2))

    // Unique layers
    val bias = AddBiasBuilder()
    val relu = ReLUBuilder()
    val lrn  = LateralResponseNormalizationBuilder(n = 5, k = 2, alpha = 1e-4f, beta = 0.75f)
    val pool = MaxPoolingBuilder(kernelPool)

    // Assemble feature extractor.
   SequenceBuilder(
      ConvolutionFilterBuilder(kernelConv11,  48), bias, relu, pool, /* lrn,*/
      ConvolutionFilterBuilder(kernelConv05, 128), bias, relu, pool, /* lrn,*/
      ConvolutionFilterBuilder(kernelConv03, 192), bias, relu,
      ConvolutionFilterBuilder(kernelConv03, 192), bias, relu,
      ConvolutionFilterBuilder(kernelConv03, 128), bias, relu, pool
    )
  }

  final def createTwoColumnFeatureExtractor()
  : ModuleBuilder = {
    val features0 = createSingleColumnFeatureExtractor()
    val features1 = features0.copy
    //features1.permuteWeightBufferReferences((b, s) => (b, s + 200))
    SequenceBuilder(
      BranchBuilder(
        features0,
        features1
      ),
      ConcatenateBuilder(TensorDomain.Sample)
    )
  }

  /**
    * http://arxiv.org/abs/1404.5997
    */
  final def createOneWeirdTrickFeatureExtractor()
  : ModuleBuilder = {
    // Kernels
    val kernelConv11 = Kernel2((11, 11), (4, 4), (2, 2))
    val kernelConv05 = Kernel2.centered((5, 5))
    val kernelConv03 = Kernel2.centered((3, 3))
    val kernelPool   = Kernel2((3, 3), (2, 2))

    // Unique layers
    val relu = ReLUBuilder()
    val bias = AddBiasBuilder()
    val pool = MaxPoolingBuilder(kernelPool)

    // Assemble feature extractor.
    SequenceBuilder(
      ConvolutionFilterBuilder(kernelConv11,  64),
      bias,
      relu,
      pool,

      ConvolutionFilterBuilder(kernelConv05, 192),
      bias,
      relu,
      pool,

      ConvolutionFilterBuilder(kernelConv03, 384),
      bias,
      relu,

      ConvolutionFilterBuilder(kernelConv03, 256),
      bias,
      relu,

      ConvolutionFilterBuilder(kernelConv03, 256),
      bias,
      relu,
      pool
    )
  }

  /**
    * http://arxiv.org/abs/1404.5997
    */
  final def createOneWeirdTrickFeatureExtractorBN()
  : ModuleBuilder = {
    // Kernels
    val kernelConv11 = Kernel2((11, 11), (4, 4), (2, 2))
    val kernelConv05 = Kernel2.centered((5, 5))
    val kernelConv03 = Kernel2.centered((3, 3))
    val kernelPool   = Kernel2((3, 3), (2, 2))

    // Unique layers
    val relu = ReLUBuilder()
    val bn   = BatchNormalizationBuilder()
    val pool = MaxPoolingBuilder(kernelPool)

    // Assemble feature extractor.
    SequenceBuilder(
      //ShowoffHistogramBuilder("Input", RealRange(-4.0f, 4.0f)).setScope(OperationScope.Channel).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      ConvolutionFilterBuilder(kernelConv11,  64),
      //ShowoffHistogramBuilder("Conv 1", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      bn,
      //ShowoffHistogramBuilder("BN 1", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      relu,
      pool,

      ConvolutionFilterBuilder(kernelConv05, 192),
      //ShowoffHistogramBuilder("Conv 2", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      bn,
      //ShowoffHistogramBuilder("BN 2", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      relu,
      pool,

      ConvolutionFilterBuilder(kernelConv03, 384),
      //ShowoffHistogramBuilder("Conv 3", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      bn,
      //ShowoffHistogramBuilder("BN 3", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      relu,

      ConvolutionFilterBuilder(kernelConv03, 256),
      //ShowoffHistogramBuilder("Conv 4", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      bn,
      //ShowoffHistogramBuilder("BN 4", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      relu,

      ConvolutionFilterBuilder(kernelConv03, 256),
      //ShowoffHistogramBuilder("Conv 5", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      bn,
      //ShowoffHistogramBuilder("BN 5", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      relu,
      pool
    )
  }

  final def createClassifier(noClasses: Int, reluThreshold: Real)
  : ModuleBuilder = {
    val dropout = DropoutBuilder(Real.pointFive)
    val relu = {
      if (reluThreshold == Real.zero) {
        ReLUBuilder()
      }
      else {
        SReLUBuilder(reluThreshold)
      }
    }

    // Assemble classifier.
    SequenceBuilder(
      ReshapeBuilder(size => Size1(1, size.noValues)),

      dropout,
      LinearFilterBuilder(4096),
      AddBiasBuilder(),
      relu,

      dropout,
      LinearFilterBuilder(4096),
      AddBiasBuilder(),
      relu,

      LinearFilterBuilder(noClasses).permuteWeightReferences(_.derive("final_filter")),
      AddBiasBuilder(),

      LogSoftmaxBuilder(),
      ClassNLLConstraintBuilder()
    )
  }

  final def createClassifierBN(noClasses: Int, reluThreshold: Real)
  : ModuleBuilder = {
    val dropout = DropoutBuilder(Real.pointFive)
    val bias    = AddBiasBuilder()
    val bn      = BatchNormalizationBuilder()
    val relu    = {
      if (reluThreshold == Real.zero) {
        ReLUBuilder()
      }
      else {
        SReLUBuilder(reluThreshold)
      }
    }

    // Assemble classifier.
    SequenceBuilder(
      ReshapeBuilder(size => Size1(1, size.noValues)),

      dropout,
      LinearFilterBuilder(4096),
      //ShowoffHistogramBuilder("FC 1", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle().setFrameWidth(1800),
      bn,
      //ShowoffHistogramBuilder("BN 6", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      relu,

      dropout,
      LinearFilterBuilder(4096),
      //ShowoffHistogramBuilder("FC 2", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      bn,
      //ShowoffHistogramBuilder("BN 7", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      relu,

      LinearFilterBuilder(noClasses).permuteWeightReferences(_.derive("final_filter")),
      //ShowoffHistogramBuilder("FC 3", RealRange(-4.0f, 4.0f)).setHandle("AX").setNormalizeMeanAndVariance(true).setDefaultFrameTitle(),
      bias,

      LogSoftmaxBuilder(),
      ClassNLLConstraintBuilder()
    )
  }

  final def createRegularizers(m: Module, lambda: ParameterBuilder)
  : Seq[RegularizerBuilder] = {
    val filters = m.weightReferences.filter(_.handle == "filter")
    val scope   = NullBuffer.derive(filters)
    Seq(
      NonEvaluatingRegularizerBuilder(
        L2WeightDecayBuilder(lambda).setBaseScope(scope)
      )
    )
  }

  final def initialize(model: Module): Unit = {
    val init0 = KaimingHeInitializerBuilder(
      InitializerGain.forReLU//,
      //GaussianDistributionBuilder(Real.zero, Real.pointFive)
    ).forReference("filter").build()
    model.reset(init0)

    val init1 = KaimingHeInitializerBuilder(
      //GaussianDistributionBuilder(Real.zero, Real.pointFive)
    ).forReference("final_filter").build()
    model.reset(init1)
  }

  final def initializeBN(model: Module): Unit = {
    val init0 = KaimingHeInitializerBuilder(
      InitializerGain.forReLU//,
      //GaussianDistributionBuilder(Real.zero, Real.one)
    ).forReference("filter").build()
    model.reset(init0)

    val init1 = XavierGlorotInitializerBuilder(
      //UniformDistributionBuilder()
    ).forReference("final_filter").build()
    model.reset(init1)

    val init2 = FixedValueInitializerBuilder.one.forReference("gamma").build()
    model.reset(init2)
  }

}
