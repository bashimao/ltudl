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

import edu.latrobe._
import edu.latrobe.blaze.{batchpools => bp}
import edu.latrobe.blaze.{objectives => obj}
import edu.latrobe.blaze.sinks.Implicits._
import edu.latrobe.blaze.sinks._
import edu.latrobe.demos.util.imagenet._
import edu.latrobe.cublaze._
import edu.latrobe.io._
import edu.latrobe.io.graph.renderers.{DotRenderer, SvgRenderer}
import edu.latrobe.time._

object TrainBlazeModel
  extends DemoEnvironment {

  val dataPath = LocalFileHandle.userHome ++ Environment.get(
    "EXPERIMENT_RELATIVE_DATA_PATH", relativeDataPathDefault
  )

  final def main(args: Array[String])
  : Unit = {
    CUBlaze.load()

    // -------------------------------------------------------------------------
    //    CREATE MODEL
    // -------------------------------------------------------------------------

    // Create model.
    val model = createModel()
    /*
    val graph = model.builder.toGraph(trainingInputHints)
    DotRenderer.render(graph, LocalFileHandle.tmp ++ "graph.dot")
    SvgRenderer.render(graph, LocalFileHandle.tmp ++ "graph.svg")
    */


    // -------------------------------------------------------------------------
    //    Load data.
    // -------------------------------------------------------------------------
    println("Loading validation set...")
    val validationSet = {
      ImageNetUtils.loadValidationSet(
        dataPath, "valid-extract-no-faulty", noClasses, cache = false
      )
    }

    println("Loading training set...")
    val trainingSet = {
      if (testMode) {
        validationSet
      }
      else {
        ImageNetUtils.loadTrainingSet(
          dataPath, "train-extract-no-faulty", noClasses, cache = false
        )
      }
    }

    println("Loading test set...")
    val testSet = {
      if (testMode) {
        validationSet
      }
      else {
        ImageNetUtils.loadTestSet(
          dataPath, "test-extract-no-faulty", cache = false
        )
      }
    }


    // -------------------------------------------------------------------------
    //    Setup preprocessing pipeline.
    // -------------------------------------------------------------------------
    val trainingInput = {
      val bp1 = createAugmentationPipelineForTraining(
        dataPath ++ "mean-and-variance-100k.json"
      )
      val bp5 = bp.AsynchronousPrefetcherBuilder(bp1)
      bp5.build(trainingLayoutHint, trainingSet)
    }

    val crossValidationPool = {
      val bp1 = createAugmentationPipelineForTraining(
        dataPath ++ "mean-and-variance-100k.json"
      )
      bp1
    }

    /*
    val ab = {
      val builders = Array(
        SequenceBuilder(
          SwitchToJVMBuilder(),
          AlexNet.createTwoColumnFeatureExtractor(),
          //AlexNet.createUnifiedFeatureExtractor(),
          AlexNet.createClassifier(noClasses, Real.zero)
          //VGG2014.createFeatureExtractorA(),
          //VGG2014.createClassifier(noClasses, Real.zero)
        ),
        SequenceBuilder(
          SwitchToCUDABuilder(),
          AlexNet.createTwoColumnFeatureExtractor(),
          //AlexNet.createUnifiedFeatureExtractor(),
          AlexNet.createClassifier(noClasses, Real.zero)
          //VGG2014.createFeatureExtractorA(),
          //VGG2014.createClassifier(noClasses, Real.zero)
        )
      )

      ArrayEx.foreach(builders)(_.permuteSeeds(seed => seed.setBaseSeed(1)))
      ArrayEx.map(builders)(_.build(trainingInputHints, InstanceSeed.zero))
    }
    ArrayEx.foreach(ab)(m => {
      m.reset(ini.HeEtAl2015Builder(
      ).setSource(
        //ini.UniformDistributionBuilder()
        ini.GaussianDistributionBuilder(
          Real.zero, Real.pointFive
        )
          .setSeed(BuilderSeed(0))).forReference("filter").build(InstanceSeed.zero))
      /*
        GaussianDistributionBuilder(
          Real.zero, 0.05f
        ).setSeed(BuilderSeed(0)).build(InstanceSeed.zero)
      )
      */
      //m.weights := 0.1f
      //_.weightsLinker.weights := 0.1f
    })
    ArrayEx.foreach(ab)(_.refresh())

    while (true) {
      val batch = trainingInput.current.get
      //val ones  = RealTensor.ones(TensorLayout(Size1(1, 4096), 1))

      val bpContexts = ArrayEx.map(ab)(
        _.predict(OperationMode.training, batch)
      )
      val out = ArrayEx.map(bpContexts)(_.output.toRealTensor)

      /*pred(1).output := pred(0).output
      pred(1).tensors.fastZip(pred(0).tensors)((a, b) => {
        if (a != null) {
          a := b
        }
      })*/

      val bbb    = out(0).valuesMatrix - out(1).valuesMatrix
      val bbbMin = breeze.linalg.min(bbb)
      val bbbMax = breeze.linalg.max(bbb)

      val out2 = ArrayEx.map(bpContexts)(_.output.toRealTensor)

      val w = ArrayEx.map(ab)(_.weights)
      val g = ArrayEx.map(w)(_.layout.toParameterBuffer(RealTensor.zeros))

      ArrayEx.foreach(ab, bpContexts, g)(
        (m, p, g) => m.deriveGradients(p, /*ones,*/ 0, g)
      )

      val ggg = g(0) - g(1)
      val gggMin = ggg.banks(0).mapSegments(_.min)
      val gggMax = ggg.banks(0).mapSegments(_.max)
      val ggggggMin = ggg.min
      val ggggggMax = ggg.max

      trainingInput.update()
    }
    */


    // -------------------------------------------------------------------------
    //    CREATE OPTIMIZER
    // -------------------------------------------------------------------------

    // Create optimizer.
    val optimizerBuilder = createOptimizer(model)

    optimizerBuilder.earlyObjectives ++= Seq(
      obj.Presets.visualizeRuntimeStatistics() >> ShowoffSinkBuilder("Runtime Stats", TimeSpan.oneMinute)
    )
    optimizerBuilder.objectives ++= Seq(
      obj.PeriodicTriggerBuilder(TimeSpan.thirtySeconds) && (obj.PrintStatusBuilder() >> ShowoffSinkBuilder("Status")),
      obj.PeriodicTriggerBuilder(TimeSpan.thirtySeconds) && obj.CrossValidationBuilder(
        testInputLayoutHint,
        validationSet,
        crossValidationPool,
        1,
        obj.Presets.visualizePerformance() >> ShowoffSinkBuilder("Cross Validation Performance", TimeSpan.oneMinute),
        obj.MultiObjectiveBuilder(
          obj.SelectTimeInSecondsBuilder() += obj.PrintValueBuilder(),
          obj.PrintStringBuilder(", "), obj.PrintIterationNoBuilder(),
          obj.PrintStringBuilder(", "), obj.PrintRunNoBuilder(),
          obj.PrintStringBuilder(", "), obj.PrintIterationNoBuilder(),
          obj.PrintStringBuilder(", "), obj.PrintValueBuilder(),
          obj.PrintStringBuilder(", "), obj.ValidateOutputBuilder.top1Label += obj.PrintValueBuilder(),
          obj.PrintStringBuilder(", "), obj.ValidateOutputBuilder.top5Label += obj.PrintValueBuilder(),
          obj.PrintStringBuilder.lineSeparator
        ) >> LocalFileHandle.userHome ++ "status.out"
      )
    )

    val optimizer = optimizerBuilder.build(model, trainingInput)

    // Perform optimization.
    //try {
    val optRes = optimizer.run()
    println(f"OptimizerResult: $optRes")
    println(f"Iterations#: ${optRes.iterationNo}, ImprovementFailures#: ${optRes.noFailures}")
  }

}
