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

package edu.latrobe.demos.mnist

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.{batchpools => bp}
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.initializers._
import edu.latrobe.blaze.optimizers._
import edu.latrobe.blaze.objectives._
import edu.latrobe.blaze.validators._
import edu.latrobe.cublaze._
import edu.latrobe.demos.util._
import edu.latrobe.kernels._
import edu.latrobe.io._

object SimpleConvNet {

  val batchSize = Environment.parseInt("EXPERIMENT_BATCH_SIZE", 128, _ > 0)

  val dataPath = Environment.get("EXPERIMENT_DATA_PATH", "./data")

  val forceCUDA = Environment.parseBoolean("EXPERIMENT_FORCE_CUDA", default = true)

  def main(args: Array[String])
  : Unit = {
    // If forced CUDA.
    if (forceCUDA) {
      CUBlaze.load()
    }

    // Load data.
    val trainingSet = MNIST.loadTrainingSet(LocalFileHandle(dataPath))
    val testSet     = MNIST.loadTestSet(LocalFileHandle(dataPath))

    // Describe pre-processing pipeline.
    val ppb0 = bp.ChooseAtRandomBuilder()
    val ppb1 = bp.MergeBuilder(ppb0, batchSize)
    val ppb2 = bp.AsynchronousPrefetcherBuilder(ppb1)
    val pp = ppb2.build(trainingSet(0).input.layout, trainingSet)

    // Describe model
    val mb = SequenceBuilder(
      if (forceCUDA) SwitchPlatformBuilder(CUDA) else IdentityBuilder(),
      // 1
      ConvolutionFilterBuilder(Kernel2((10, 10)), 20),
      AddBiasBuilder(),
      TanhBuilder(),
      // 2
      MaxPoolingBuilder(Kernel2((2, 2), (2, 2))),
      // 3
      ConvolutionFilterBuilder(Kernel2((5, 5)), 40),
      AddBiasBuilder(),
      TanhBuilder(),
      // 2
      MaxPoolingBuilder(Kernel2((3, 3), (3, 3))),
      // 4
      LinearFilterBuilder(150),
      AddBiasBuilder(),
      TanhBuilder(),
      // 5
      ReshapeBuilder.collapseDimensions(),
      // 6
      LinearFilterBuilder(10),
      AddBiasBuilder(),
      LogSoftmaxBuilder(),
      // 7
      ClassNLLConstraintBuilder()
    )
    val m = mb.build(pp.outputHints)

    // Initialize model.
    val ini = XavierGlorotInitializerBuilder().build()
    m.reset(ini)

    // Describe optimizer.
    val ob = AdaDeltaBuilder()
    ob.objectives += IterationCountLimitBuilder(2500L)
    ob.objectives += CyclicTriggerBuilder(50) && CrossValidationBuilder(
      testSet(0).input.layout,
      testSet,
      ppb2,
      1,
      PrintStatusBuilder(
        (Top1LabelValidatorBuilder(), (x: ValidationScore) => f"Top1 ${x.accuracy * 100}%.1f %%")
      ),
      PrintStringBuilder.lineSeparator
    )

    // Run optimizer.
    val o = ob.build(m, pp)
    o.run()


    // Online cross validation is nice. Here is how you do offline CV.

    // Describe pre-processing pipeline.
    val testPPB0 = bp.ConsumeInOrderBuilder()
    val testPPB1 = bp.MergeBuilder(testPPB0, batchSize)
    val testPP = testPPB1.build(testSet(0).input.layout, testSet)

    // Create validator
    val validator = Top1LabelValidatorBuilder().build()

    // Run validator
    val score = validator.apply(m, testPP)
    println(f"Test Set Accuracy: ${score.accuracy * 100}%.1f %%")
  }

}
