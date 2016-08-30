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

package edu.latrobe.demos.imagenet

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.initializers._
import edu.latrobe.blaze.modules._
import edu.latrobe.kernels._
import edu.latrobe.sizes.Size1

object ResNet {

  private var shortcutType = 'B'

  private var noMapsPrev = -1

  final def createShortcut(stride: (Int, Int),
                           noMaps: Int)
  : ModuleBuilder = shortcutType match {
    case 'A' =>
      if (noMaps != noMapsPrev) {
        createShortcutA(stride, noMaps)
      }
      else {
        IdentityBuilder()
      }

    case 'B' =>
      if (noMaps != noMapsPrev) {
        createShortcutB(stride, noMaps)
      }
      else {
        IdentityBuilder()
      }

    case 'C' =>
      createShortcutB(stride, noMaps)

    case _ =>
      throw new MatchError(shortcutType)
  }

  final def createShortcutA(stride: (Int, Int),
                            noMaps: Int)
  : SequenceBuilder = SequenceBuilder(
    MeanPoolingBuilder(Kernel2((1, 1), stride)),
    BranchBuilder(
      IdentityBuilder(),
      MultiplyValuesBuilder(TensorDomain.Batch, Real.zero)
    ),
    ConcatenateBuilder(TensorDomain.Channel)
  ).setHandle("Shortcut A")

  final def createShortcutB(stride: (Int, Int),
                            noMaps: Int)
  : SequenceBuilder = SequenceBuilder(
    ConvolutionFilterBuilder(Kernel2((1, 1), stride), noMaps),
    BatchNormalizationBuilder()
  ).setHandle("Shortcut B")

  final def createBasicBlock(stride: (Int, Int),
                             noMaps: Int)
  : SequenceBuilder = {
    val filter = SequenceBuilder(
      ConvolutionFilterBuilder(Kernel2((3, 3), stride, (1, 1)), noMaps),
      BatchNormalizationBuilder(),
      ReLUBuilder(),
      ConvolutionFilterBuilder(Kernel2((3, 3), (1, 1), (1, 1)), noMaps),
      BatchNormalizationBuilder()
    )

    val shortcut = createShortcut(stride, noMaps)

    val block = SequenceBuilder(
      BranchBuilder(filter, shortcut),
      MergeBuilder(),
      ReLUBuilder()
    ).setHandle("Basic Block")

    noMapsPrev = noMaps
    block
  }

  final def createBottleneckBlock(stride:   (Int, Int),
                                  noMaps:   Int)
  : SequenceBuilder = {
    val noMaps4 = noMaps * 4

    val filter = SequenceBuilder(
      ConvolutionFilterBuilder(Kernel2((1, 1), (1, 1), (0, 0)), noMaps),
      BatchNormalizationBuilder(),
      ReLUBuilder(),
      ConvolutionFilterBuilder(Kernel2((3, 3), stride, (1, 1)), noMaps),
      BatchNormalizationBuilder(),
      ReLUBuilder(),
      ConvolutionFilterBuilder(Kernel2((1, 1), (1, 1), (0, 0)), noMaps4),
      BatchNormalizationBuilder()
    )

    val shortcut = createShortcut(stride, noMaps4)

    val block = SequenceBuilder(
      BranchBuilder(filter, shortcut),
      MergeBuilder(),
      ReLUBuilder()
    ).setHandle("Bottleneck Block")

    noMapsPrev = noMaps4
    block
  }

  final def createBlockGroup(noBlocks:      Int,
                             stride:        (Int, Int),
                             noMaps:        Int,
                             createBlockFn: ((Int, Int), Int) => ModuleBuilder)
  : ModuleBuilder = {
    val group = SequenceBuilder()

    // Translation block.
    group += createBlockFn(stride, noMaps)

    // Mapping blocks.
    for (i <- 1 until noBlocks) {
      group += createBlockFn((1, 1), noMaps)
    }

    group
  }


  final def createForImageNet(depth: Int, noClasses: Int): SequenceBuilder = {
    val config = Map(
       18 -> Array(
         (2, (1, 1),  64, createBasicBlock(_, _)),
         (2, (2, 2), 128, createBasicBlock(_, _)),
         (2, (2, 2), 256, createBasicBlock(_, _)),
         (2, (2, 2), 512, createBasicBlock(_, _))
       ),
       34 -> Array(
         (3, (1, 1),  64, createBasicBlock(_, _)),
         (4, (2, 2), 128, createBasicBlock(_, _)),
         (6, (2, 2), 256, createBasicBlock(_, _)),
         (3, (2, 2), 512, createBasicBlock(_, _))
       ),
       50 -> Array(
         (3, (1, 1),  64, createBottleneckBlock(_, _)),
         (4, (2, 2), 128, createBottleneckBlock(_, _)),
         (6, (2, 2), 256, createBottleneckBlock(_, _)),
         (3, (2, 2), 512, createBottleneckBlock(_, _))
       ),
      101 -> Array(
        ( 3, (1, 1),  64, createBottleneckBlock(_, _)),
        ( 4, (2, 2), 128, createBottleneckBlock(_, _)),
        (23, (2, 2), 256, createBottleneckBlock(_, _)),
        ( 3, (2, 2), 512, createBottleneckBlock(_, _))
      ),
      152 -> Array(
        ( 3, (1, 1),  64, createBottleneckBlock(_, _)),
        ( 8, (2, 2), 128, createBottleneckBlock(_, _)),
        (36, (2, 2), 256, createBottleneckBlock(_, _)),
        ( 3, (2, 2), 512, createBottleneckBlock(_, _))
      )
    )

    val seq = SequenceBuilder()
    seq += ConvolutionFilterBuilder(Kernel2((7, 7), (2, 2), (3, 3)), 64)
    seq += BatchNormalizationBuilder()
    seq += ReLUBuilder()
    seq += MaxPoolingBuilder(Kernel2((3, 3), (2, 2), (1, 1)))
    val c = config(depth)
    noMapsPrev = c(0)._3
    ArrayEx.foreach(c)({
      case (noBlocks, stride, noMaps, createBlockFn) =>
        seq += createBlockGroup(noBlocks, stride, noMaps, createBlockFn)
    })
    seq += MeanPoolingBuilder(Kernel2((7, 7), (1, 1)))
    seq += ReshapeBuilder.collapseDimensions()
    seq += LinearFilterBuilder(noClasses).permuteWeightReferences(_.derive("final_filter"))
    seq += AddBiasBuilder()
    seq += LogSoftmaxBuilder()
    seq += ClassNLLConstraintBuilder()
    seq
  }

  final def initialize(model: Module): Unit = {
    val init0 = KaimingHeInitializerBuilder(
      InitializerGain.forReLU
      //GaussianDistributionBuilder(Real.zero, Real.pointFive)
      //GaussianDistributionBuilder(Real.zero, Real.one)
    ).setUseOutputFanSize(true).forReference("filter").build()
    model.reset(init0)

    /*
    val init0 = XavierBuilder(
      UniformDistributionBuilder()
    ).forReference("filter").build()
    model.reset(init0)
    */
/*
    val init1 = XavierGlorotInitializerBuilder(
      //UniformDistributionBuilder()
      //GaussianDistributionBuilder(Real.zero, Real.one)
    ).forReference("final_filter").build()
    */
    val init1 = KaimingHeInitializerBuilder(
      InitializerGain.forSigmoid//,
      //UniformDistributionBuilder(-1.0f, 1.0f)
    ).forReference("final_filter").build()
    model.reset(init1)

    val init2 = FixedValueInitializerBuilder.one.forReference("gamma").build()
    model.reset(init2)
  }

}
