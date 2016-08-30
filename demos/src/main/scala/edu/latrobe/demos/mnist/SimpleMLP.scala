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

package edu.latrobe.demos.mnist

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.{batchpools => bp}
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.initializers._
import edu.latrobe.blaze.optimizers._
import edu.latrobe.blaze.objectives._
import edu.latrobe.blaze.validators._
import edu.latrobe.demos.util._
import edu.latrobe.io._

object SimpleMLP {

  val batchSize = Environment.parseInt("EXPERIMENT_BATCH_SIZE", 128, _ > 0)

  val dataPath = Environment.get("EXPERIMENT_DATA_PATH", "./data")

  def main(args: Array[String]): Unit = {
    // Load data.
    val trainingSet = MNIST.loadTrainingSet(LocalFileHandle(dataPath))
    val testSet     = MNIST.loadTestSet(LocalFileHandle(dataPath))

    // Describe pre-processing pipeline.
    val ppb0 = bp.ChooseAtRandomBuilder()
    val ppb1 = bp.MergeBuilder(ppb0, batchSize)
    val ppb2 = bp.BatchAugmenterBuilder(ppb1,
      ReshapeBuilder.collapseDimensions()
    )
    val ppb3 = bp.AsynchronousPrefetcherBuilder(ppb2)
    val pp = ppb3.build(trainingSet(0).input.layout, trainingSet)

    // Describe model
    val mb = SequenceBuilder(
      // 1
      LinearFilterBuilder(50),
      AddBiasBuilder(),
      TanhBuilder(),
      // 2
      LinearFilterBuilder(10),
      AddBiasBuilder(),
      LogSoftmaxBuilder(),
      // 3
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
    val testPPB2 = bp.BatchAugmenterBuilder(testPPB1,
      ReshapeBuilder.collapseDimensions()
    )
    val testPP = testPPB2.build(testSet(0).input.layout, testSet)

    // Create validator
    val validator = Top1LabelValidatorBuilder().build()

    // Run validator
    val score = validator.apply(m, testPP)
    println(f"Test Set Accuracy: ${score.accuracy * 100}%.1f %%")
  }

}
