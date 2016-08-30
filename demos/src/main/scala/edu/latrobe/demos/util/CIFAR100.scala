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

package edu.latrobe.demos.util

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.sizes._
import edu.latrobe.io._
import java.io.{BufferedInputStream, DataInputStream}
import scala.collection.parallel.mutable._
import scala.util.matching._

/**
  * Some functions for handling the CIFAR100 dataset.
  */
object CIFAR100 {

  final private val size = Size2(32, 32, 3)

  final private def loadBinFile(stream: DataInputStream)
  : Array[(Array[Byte], Int, Int)] = {
    val r       = new Array[Byte](size.noTuples)
    val g       = new Array[Byte](size.noTuples)
    val b       = new Array[Byte](size.noTuples)
    val builder = Array.newBuilder[(Array[Byte], Int, Int)]
    while (true) {
      val coarseLabel = stream.read()
      if (coarseLabel < 0) {
        return builder.result()
      }

      val fineLabel = stream.readByte.toInt

      assume(stream.read(r) == size.noTuples)
      assume(stream.read(g) == size.noTuples)
      assume(stream.read(b) == size.noTuples)
      val bytes = new Array[Byte](size.noValues)
      ArrayEx.interleave(bytes, r, g, b)

      builder += Tuple3(bytes, coarseLabel, fineLabel)
    }
    throw new UnknownError
  }

  final private def loadBinFile(file: FileHandle)
  : Array[(Array[Byte], Int, Int)] = {
    using(
      new DataInputStream(
        new BufferedInputStream(
          file.openStream()
        )
      )
    )(loadBinFile)
  }

  final private def load(file:          FileHandle,
                         filter:        Regex,
                         useFineLabels: Boolean)
  : Array[Batch] = {
    val coarseClassLabels = {
      val lines = (file ++ "coarse_label_names.txt").readLines()
      ArrayEx.filter(lines)(_.length > 0)
    }
    val fineClassLabels = {
      val lines = (file ++ "fine_label_names.txt").readLines()
      ArrayEx.filter(lines)(_.length > 0)
    }

    val files = ParArray.handoff(
      file.listFiles((depth, handle) => handle.matches(filter))
    )
    val pairs = files.flatMap(loadBinFile(_))


    val noClasses = {
      if (useFineLabels) {
        fineClassLabels.length
      }
      else {
        coarseClassLabels.length
      }
    }
    pairs.map(kv => {
      val classNo = if (useFineLabels) kv._3 else kv._2
      Batch(
        ByteArrayTensor.derive(size, kv._1),
        SparseRealMatrixTensor(MatrixEx.labelsToSparse(noClasses, classNo)),
        Array((coarseClassLabels(kv._2), fineClassLabels(kv._3)))
      )
    }).toArray
  }

  final def loadTrainingSet(file:          FileHandle,
                            useFineLabels: Boolean = true)
  : Array[Batch] = load(file, "train\\.bin$".r, useFineLabels)

  final def loadTestSet(file:          FileHandle,
                        useFineLabels: Boolean = true)
  : Array[Batch] = load(file, "test\\.bin$".r, useFineLabels)

}
