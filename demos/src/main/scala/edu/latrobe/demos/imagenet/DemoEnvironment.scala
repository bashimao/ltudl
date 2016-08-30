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
import edu.latrobe.blaze.{batchpools => bp}
import edu.latrobe.blaze.initializers._
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.modules.{bitmap => bmp}
import edu.latrobe.blaze.{objectives => obj}
import edu.latrobe.blaze.optimizers._
import edu.latrobe.blaze.parameters._
import edu.latrobe.blaze.{sinks => snk}
import edu.latrobe.cublaze._
import edu.latrobe.demos.util.imagenet.ImageNetUtils
import edu.latrobe.io.FileHandle
import edu.latrobe.io.image.BitmapFormat
import edu.latrobe.sizes._
import edu.latrobe.time.TimeSpan

class DemoEnvironment {

  final val experimentName
  : String = Environment.get("EXPERIMENT_NAME", "no name given")

  final val relativeDataPathDefault
  : String = "Share/Datasets/ImageNet/CLSLOC"

  final val relativeModelPathDefault
  : String = "Share/Models/ImageNet/backup.model"

  //val trainingBatchSize = 16
  //val trainingBatchSize = 32
  //val trainingBatchSize = 64
  //val trainingBatchSize = 96
  //val trainingBatchSize = 128
  //val trainingBatchSize = 512
  final val trainingBatchSize = Environment.parseInt("EXPERIMENT_TRAINING_BATCH_SIZE", 128, _ > 0)

  final val testBatchSize = Environment.parseInt("EXPERIMENT_TEST_BATCH_SIZE", 100, x => x > 0 && x % 10 == 0)

  //val noClasses = 10
  //val noClasses = 100
  //val noClasses = 1000
  final val noClasses = Environment.parseInt("EXPERIMENT_NO_CLASSES", 1000, _ > 0)

  //val modelName = "ResNet-18"
  //val modelName = "ResNet-18-PreAct"
  //val modelName = "ResNet-34"
  //val modelName = "ResNet-50"
  //val modelName = "ResNet-101"
  //val modelName = "ResNet-101-PreAct"
  //val modelName = "ResNet-152"
  //val modelName = "ResNet-152-PreAct"
  //val modelName = "AlexNet"
  //val modelName = "AlexNet-Uni"
  //val modelName = "AlexNet-OWT"
  //val modelName = "AlexNet-OWT-BN"
  //val modelName = "VGG-A"
  final val modelName = Environment.get("EXPERIMENT_MODEL_NAME", "ResNet-18")

  // Use fixed size for training.
  final val scaleSize = Size2(256, 256, 3)
  //val scaleSize = 32

  final val trainingLayoutHint = IndependentTensorLayout(scaleSize,          trainingBatchSize)
  final val trainingInputSize = Size2(224, 224, 3)
  //val trainingInputSize = Size2((4, 4), 3)
  //val trainingInputSize = Size2((32, 32), 3)
  final val trainingInputLayout = IndependentTensorLayout(trainingInputSize, trainingBatchSize)
  final val trainingInputHints  = BuildHints(JVM, trainingInputLayout)

  final val testLayoutHint      = IndependentTensorLayout(scaleSize,         testBatchSize)
  final val testInputLayoutHint = IndependentTensorLayout(trainingInputSize, testBatchSize)

  final val testMode = Environment.parseBoolean("EXPERIMENT_TEST_MODE", default = false)

  final def createModel()
  : Module = {
    // Create
    val modBuilder = modelName match {
      case "AlexNet" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          AlexNet.createTwoColumnFeatureExtractor(),
          AlexNet.createClassifier(noClasses, Real.zero)
        )

      case "AlexNet-Uni" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          AlexNet.createUnifiedFeatureExtractor(),
          AlexNet.createClassifier(noClasses, Real.zero)
        )

      case "AlexNet-OWT" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          AlexNet.createOneWeirdTrickFeatureExtractor(),
          AlexNet.createClassifier(noClasses, Real.zero)
        )

      case "AlexNet-OWT-BN" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          AlexNet.createOneWeirdTrickFeatureExtractorBN(),
          AlexNet.createClassifierBN(noClasses, Real.zero)
        )

      case "VGG-A" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          VGG2014.createFeatureExtractorA(),
          VGG2014.createClassifier(noClasses, Real.zero)
        )

      case "ResNet-18" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          ResNet.createForImageNet(18, noClasses)
        )

      case "ResNet-18-PreAct" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          PreActResNet.createForImageNet(18, noClasses)
        )

      case "ResNet-34" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          ResNet.createForImageNet(34, noClasses)
        )

      case "ResNet-34-PreAct" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          PreActResNet.createForImageNet(34, noClasses)
        )

      case "ResNet-50" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          ResNet.createForImageNet(50, noClasses)
        )

      case "ResNet-50-PreAct" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          PreActResNet.createForImageNet(50, noClasses)
        )

      case "ResNet-101" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          ResNet.createForImageNet(101, noClasses)
        )

      case "ResNet-101-PreAct" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          PreActResNet.createForImageNet(101, noClasses)
        )

      case "ResNet-152" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          ResNet.createForImageNet(152, noClasses)
        )

      case "ResNet-152-PreAct" =>
        SequenceBuilder(
          SwitchPlatformBuilder(CUDA),
          PreActResNet.createForImageNet(152, noClasses)
        )

      case _ =>
        throw new MatchError(modelName)
    }

    // TODO: Just for me having a comfortable breakpoint. Can safely remove this.
    val b2 = modBuilder.copy
    val dummy = {
      if (b2 == modBuilder) {
        println("Copy successful! 1")
      }
      else {
        println("Copy unsuccessful! 0")
      }
    }

    val model = modBuilder.build(trainingInputHints)

    // Initialize
    if (modelName.startsWith("ResNet")) {
      ResNet.initialize(model)
    }
    else if (modelName.startsWith("AlexNet")) {
      if (modelName.contains("BN")) {
        AlexNet.initializeBN(model)
      }
      else {
        AlexNet.initialize(model)
      }
    }
    else {
      model.reset(
        KaimingHeInitializerBuilder(
          //ini.UniformDistributionBuilder()
          //ini.GaussianDistributionBuilder(Real.zero, Real.pointFive)
        ).forReference("filter").build()
      )
    }

    model
  }

  final def createAugmentationPipelineForTraining(mvPath: FileHandle)
  : BatchPoolBuilder = {
    val bp0 = bp.ChooseAtRandomBuilder()
    val bp1 = bp.MergeBuilder(bp0, trainingBatchSize)
    val bp2 = bp.BatchAugmenterBuilder(
      bp1,
      SequenceBuilder(
        raw.DecodeBitmapsBuilder(),
        bmp.ResampleExBuilder.fixShortEdge(
          scaleSize.dims._1, BitmapFormat.BGR
        ),
        bmp.CropRandomPatchBuilder(trainingInputSize.dims)
      )
    )
    val bp3 = bp.SampleAugmenterBuilder(
      bp2,
      RandomPathBuilder(
        IdentityBuilder(),
        bmp.FlipHorizontalBuilder()
      )
    )

    val mvs = ImageNetUtils.loadMeanAndVariance(mvPath)
    bp.BatchAugmenterBuilder(
      bp3,
      SequenceBuilder(
        bmp.ConvertToRealBuilder(),
        AddValuesBuilder(
          TensorDomain.Channel, ArrayEx.map(mvs)(Real.zero - _.mean)
        ),
        MultiplyValuesBuilder(
          TensorDomain.Channel, ArrayEx.map(mvs)(_.sampleStdDevInv())
        )
      )
    )
  }

  final def createAugmentationPipelineForValidation(mvPath: FileHandle)
  : BatchPoolBuilder = {
    // Note that the "number of repetitions" are related here because we want
    // 10 augmentations of the same sample.
    // Sequence:
    // 2 x (center, top left, top right, bottom left, bottom right)
    // 5 x (no flip), 5 x flip
    val bp0 = bp.ConsumeInOrderBuilder()
    val bp1 = bp.RepeatBuilder(bp0, 10)
    val bp2 = bp.MergeBuilder(bp1, testBatchSize / 10)
    val bp3 = bp.BatchAugmenterBuilder(
      bp2,
      SequenceBuilder(
        raw.DecodeBitmapsBuilder(),
        bmp.CropCenterSquareBuilder(),
        bmp.ResampleBuilder(scaleSize.dims, BitmapFormat.BGR)
      )
    )
    // 10-crop
    val bp4 = bp.SampleAugmenterBuilder(
      bp3,
      SequenceBuilder(
        AlternatePathBuilder(
          bmp.CropCenterPatchBuilder(trainingInputSize.dims), // center
          bmp.CropPatchBuilder(trainingInputSize.dims, (0,                          0)), // top left
          bmp.CropPatchBuilder(trainingInputSize.dims, (-trainingInputSize.dims._1, 0)), // top right
          bmp.CropPatchBuilder(trainingInputSize.dims, (0,                          -trainingInputSize.dims._2)), // bottom left
          bmp.CropPatchBuilder(trainingInputSize.dims, (-trainingInputSize.dims._1, -trainingInputSize.dims._2)) // bottom right
        ),
        AlternatePathBuilder(
          IdentityBuilder(),
          bmp.FlipHorizontalBuilder()
        )
      )
    )

    val mvs = ImageNetUtils.loadMeanAndVariance(mvPath)

    val bp5 = bp.BatchAugmenterBuilder(
      bp4,
      SequenceBuilder(
        bmp.ConvertToRealBuilder(),
        AddValuesBuilder(
          TensorDomain.Channel, ArrayEx.map(mvs)(Real.zero - _.mean)
        ),
        MultiplyValuesBuilder(
          TensorDomain.Channel, ArrayEx.map(mvs)(_.sampleStdDevInv())
        )
      )
    )

    bp5
  }

  final def createOptimizer(model: Module)
  : OptimizerBuilder = {
    // Create model specific regularizers.
    val regBuilders = {
      val lambda = ConstantValueBuilder(5e-4f)
      /*
      L2RegularizerBuilder(5e-4f) ++= Array(
        (0, 10), (0, 20), (0, 30), (0, 40),
        (0, 50), (0, 60), (0, 70), (0, 80)
      )
      */
      modelName match {
        case "AlexNet" =>
          AlexNet.createRegularizers(model, lambda)
        case "AlexNet-Uni" =>
          AlexNet.createRegularizers(model, lambda)
        case "AlexNet-OWT" =>
          AlexNet.createRegularizers(model, lambda)
        case _ =>
          Seq.empty
      }
      /*
      .setWeightDecayRateL2(
        DiscreteStepsBuilder(
          (    0L, 5e-4f),
          (60000L, Real.zero)
        )
      )
      */
    }

    // Create optimizer.
    val builder = AdaDeltaBuilder()
    //builder.setEpsilon(1e-1f)
    //val builder = RMSPropBuilder()
    /*
    val localOptimizer = MomentumBuilder().setLearningRate(
      par.ConstantValueBuilder(1e-2f)
      //par.ConstantValueBuilder(1e-2f)
      //par.ConstantValueBuilder(1e-1f)
    )
    */
    /*
    val builder = MomentumBuilder().setLearningRate(
      DiscreteStepsBuilder(
        (    0L, 1e-2f),
        (20000L, 5e-3f),
        (40000L, 1e-3f),
        (60000L, 5e-4f),
        (80000L, 1e-4f)
      )
    )
    */
    /*
    val localOptimizer = TraditionalSGDBuilder().setLearningRate(
      par.ConstantValueBuilder(1e-2f)
    )
    */
    /*
    val localOptimizer = GradientDescentBuilder().setLearningRate(
      par.ConstantValueBuilder(1e-2f)
    )
    */
    // val builder = AdamBuilder().setLearningRate(RealSeriesBuilder.derive(1e-5f))
    builder.regularizers ++= regBuilders


    builder.objectives ++= Seq(
      obj.PrintStatusBuilder() || obj.PrintStringBuilder.lineSeparator, // Print current optimizer state to stderr.

      obj.Presets.visualizePerformance() >> snk.ShowoffSinkBuilder("Performance", TimeSpan.thirtySeconds)//,
      /*
      ShowoffPerformanceBuilder(TimeSpan.thirtySeconds) ++= Seq(
        ("Top-1 Acc", top1, (vr: ValidationScore) => vr.accuracy * 100),
        ("Top-5 Acc", top5, (vr: ValidationScore) => vr.accuracy * 100)
      )*/
      /*
      ShowoffHyperParametersBuilder(TimeSpan.oneMinute) ++= Seq(
        ("LR", 0),
        ("WDecayL2", 1)
      ),
      ShowoffHyperParametersBuilder(TimeSpan.oneMinute) ++= Seq(
        ("VDecay", 0),
        ("Dampening", 0)
      ),
      */
      //PeriodicTriggerBuilder(TimeSpan.tenSeconds) && (PrintStatusBuilder() >> ShowoffSinkBuilder("Status"))
      //PeriodicTriggerBuilder(TimeSpan.thirtySeconds),
      //CyclicTriggerBuilder(50L)
      /*PeriodicTriggerBuilder(TimeSpan.oneMinute) && CrossValidationBuilder(
        trainingInputLayout,
        validationSet.collect(),
        crossValidationPool,
        1,
        VisualizeCurvesBuilder.performance() >> ShowoffSinkBuilder("Cross Validation Performance")
      )*/

      // Actual limits.
      //ValueLimitBuilder(Real.negativeInfinity, 0.00001f), // If regret/cost falls below limit.
      //RunIterationCountLimitBuilder(10000000), // Or 500 iterations.
      //TimeLimitBuilder(TimeSpan.threeDays) // Or training time exceeds 3 days.
    )

    // TODO: Just for testing. Can remove this safely.
    val b2 = builder.copy
    val dummy = {
      if (b2 == builder) {
        println("Optimizer copy successful! 1")
      }
      else {
        println("Optimizer copy failed! 0")
      }
    }

    builder
  }

}
