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
import edu.latrobe.blaze.batchpools.{ChooseAtRandomBuilder, _}
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.modules.{bitmap => bmp}
import edu.latrobe.sizes._
import edu.latrobe.demos.util.imagenet._
import edu.latrobe.io._
import edu.latrobe.io.image._
import edu.latrobe.time._
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods

object ComputeMeanAndVariance {

  val dataFile = LocalFileHandle.userHome ++ "Share" ++ "Datasets" ++ "ImageNet" ++ "CLSLOC"

  val backupFile = LocalFileHandle.root ++ "tmp" ++ "meanAndVariance.json.backup"

  val resultFile = LocalFileHandle.root ++ "tmp" ++ "meanAndVariance.json"

  val scaleSize = 256

  val noClasses = 1000

  final def main(args: Array[String]): Unit = {
    val inputHints = BuildHints(JVM, IndependentTensorLayout(Size2(256, 256, 3), 128))

    println("Loading training set...")
    val trainingSet = ImageNetUtils.loadTrainingSet(
      dataFile, "train-extract-no-faulty", noClasses, cache = false
    )

    println("Creating augmentation pipeline...")
    val batches = {
      val builder = BatchAugmenterBuilder(
        ChooseAtRandomBuilder(),
        SequenceBuilder(
          bmp.ResampleExBuilder.fixShortEdge(scaleSize, BitmapFormat.BGR),
          bmp.ConvertToRealBuilder()
        )
      )
      builder.build(inputHints.layout.makeIndependent, trainingSet)
    }

    println("Creating modules...")
    val module: Module = {
      val builder = IdentityBuilder()
      builder.build(inputHints)
    }
   val mvs: Array[MeanAndVariance] = Array.fill[MeanAndVariance](
     inputHints.layout.size.noChannels
   )(MeanAndVariance())

    println("Computing mean and variance...")
    var prevNoImages = 0
    var prevNoPixels = 0L
    var prevNow      = Timestamp.now()
    var noImages     = 0
    while (System.in.available == 0) {
      val drawContext = batches.draw()
      val batch       = drawContext.batch
      batch.tags(0) match {
        case (fileName, _, _) =>
          logger.trace(f"Processing: $fileName")
      }
      val result = module.predict(
        Training(0L), batch
      ).dropIntermediates()

      var i = 0
      MatrixEx.foreach(result.output.valuesMatrix)(value => {
        mvs(i).update(value)
        i = (i + 1) % mvs.length
      })

      val now = Timestamp.now()
      val diff = TimeSpan(prevNow, now).seconds
      if (diff >= 5.0) {
        println(
          f"Mean: ${mvs(0).mean}%.3f ${mvs(1).mean}%.3f ${mvs(2).mean}%.3f, " +
          f"StdDev: ${mvs(0).sampleStdDev()}%.3f ${mvs(1).sampleStdDev()}%.3f ${mvs(2).sampleStdDev()}%.3f, " +
          f"$noImages%d images viewed (${(noImages - prevNoImages) / diff}%.3f / s), "
        )
        prevNoImages = noImages
        prevNow      = now

        val json = JObject(
          Json.field("r", mvs(2)),
          Json.field("g", mvs(1)),
          Json.field("b", mvs(0))
        )
        backupFile.writeText(JsonMethods.compact(json))
      }
      noImages += 1

      drawContext.close()
    }

    println("Saving results...")
    val json = JObject(
      Json.field("r", mvs(2)),
      Json.field("g", mvs(1)),
      Json.field("b", mvs(0))
    )
    resultFile.writeText(JsonMethods.compact(json))
  }

}
