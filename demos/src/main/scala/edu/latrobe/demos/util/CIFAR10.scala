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
 * Some functions for handling the CIFAR10 dataset.
 */
object CIFAR10 {

  final private val size = Size2(32, 32, 3)

  final private def loadBinFile(stream: DataInputStream)
  : Array[(Array[Byte], Int)] = {
    val r       = new Array[Byte](size.noTuples)
    val g       = new Array[Byte](size.noTuples)
    val b       = new Array[Byte](size.noTuples)
    val builder = Array.newBuilder[(Array[Byte], Int)]
    while (true) {
      val label = stream.read()
      if (label < 0) {
        return builder.result()
      }

      assume(stream.read(r) == size.noTuples)
      assume(stream.read(g) == size.noTuples)
      assume(stream.read(b) == size.noTuples)
      val bytes = new Array[Byte](size.noValues)
      ArrayEx.interleave(bytes, r, g, b)

      builder += Tuple2(bytes, label)
    }
    throw new UnknownError
  }

  final private def loadBinFile(file: FileHandle)
  : Array[(Array[Byte], Int)] = {
    using(
      new DataInputStream(
        new BufferedInputStream(
          file.openStream()
        )
      )
    )(loadBinFile)
  }

  final private def load(file:   FileHandle,
                         filter: Regex)
  : Array[Batch] = {
    val classLabels = {
      val lines = (file ++ "batches.meta.txt").readLines()
      ArrayEx.filter(lines)(_.length > 0)
    }

    val files = ParArray.handoff(
      file.listFiles((depth, handle) => handle.matches(filter))
    )
    val pairs = files.flatMap(loadBinFile(_))

    pairs.map(kv => Batch(
      ByteArrayTensor.derive(size, kv._1),
      SparseRealMatrixTensor(MatrixEx.labelsToSparse(classLabels.length, kv._2)),
      Array(classLabels(kv._2))
    )).toArray
  }

  final def loadTrainingSet(file: FileHandle)
  : Array[Batch] = load(file, "data_batch_.\\.bin$".r)

  final def loadTestSet(file: FileHandle)
  : Array[Batch] = load(file, "test_batch\\.bin$".r)

}
