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
import edu.latrobe.demos.util.imagenet._
import edu.latrobe.blaze.validators._
import edu.latrobe.cublaze._
import edu.latrobe.io._
import edu.latrobe.time._
import java.io.ObjectInputStream

object TestBlazeModel
  extends DemoEnvironment {

  val modelPath = LocalFileHandle.userHome ++ Environment.get(
    "TEST_RELATIVE_MODEL_PATH", relativeModelPathDefault
  )

  val dataPath = LocalFileHandle.userHome ++ Environment.get(
    "TEST_RELATIVE_DATA_PATH", relativeDataPathDefault
  )

  final def main(args: Array[String])
  : Unit = {
    CUBlaze.load()

    // -------------------------------------------------------------------------
    //    CREATE MODEL
    // -------------------------------------------------------------------------

    // Create model.
    val model = createModel()

    // Load old parameters.
    val oldParams = {
      using(new ObjectInputStream(modelPath.openStream()))(stream => {
        val tmp = stream.readObject()
        tmp.asInstanceOf[ValueTensorBuffer]
      })
    }
    model.weightBuffer := oldParams


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
    //    OFFLINE VALIDATION
    // -------------------------------------------------------------------------
    val validationTestInput = {
      val bp0 = TrainBlazeModel.createAugmentationPipelineForValidation(
        dataPath ++ "mean-and-variance-100k.json"
      )
      val bp1 = bp.AsynchronousPrefetcherBuilder(bp0)
      bp1.build(testInputLayoutHint, validationSet)
    }

    // TODO: There is one minor issue here. We do not score per-sample but per
    // TODO: augmentation. So outputting the raw counter values could be
    // TODO: misleading. However, the "accuracy" is a relative measure. So,
    // TODO: it should still give you a good clue how good the model is.

    // Test model.
    val validator1   = Top1LabelValidatorBuilder().build()
    val validator5   = TopKLabelsValidatorBuilder(5).build()
    var vr1          = ValidationScore.zero
    var vr5          = ValidationScore.zero
    var noTestedPrev = 0L
    var noTested     = 0L
    val timestamp0   = Timestamp.now()
    var nowPrev      = timestamp0
    validationTestInput.foreach(
      batch => using(
        model.predict(Training(0L), batch)
      )(p => {
        val now = Timestamp.now()

        vr1      += validator1.apply(p.reference, p.output)
        vr5      += validator5.apply(p.reference, p.output)
        noTested += testBatchSize / 10

        val total = TimeSpan(timestamp0, now).seconds
        val diff  = TimeSpan(nowPrev, now).seconds
        println(f"$total%7.3f [$diff%+6.3f] Top1: ${vr1.accuracy * 100}%5.1f%%, Top5: ${vr5.accuracy * 100}%5.1f%% $noTested images tested! (${(noTested - noTestedPrev) / diff} images/s)")
        noTestedPrev = noTested
        nowPrev      = now
      })
    )

    println(f"Final Top1: ${vr1.accuracy * 100}%5.1f%%, Top5: ${vr5.accuracy * 100}%5.1f%%")
  }

}
