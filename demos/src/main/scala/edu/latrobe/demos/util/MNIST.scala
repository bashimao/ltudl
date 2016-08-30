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

package edu.latrobe.demos.util

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.sizes._
import edu.latrobe.io._
import java.io._

object MNIST {

  final private def loadImages(stream: DataInputStream)
  : Array[RealArrayTensor] = {
    // Check magic number.
    val magic = stream.readInt
    assume(magic == 2051)

    // Get size of dataset.
    val noImages = stream.readInt
    assume(noImages > 0)
    val width = stream.readInt
    assume(width > 0)
    val height = stream.readInt
    assume(height > 0)

    // Compute size.
    val size = Size2(width, height, 1)

    // Extract individual arrays for each bitmap and return.
    val factor = Real.one / 255
    val result = new Array[RealArrayTensor](noImages)
    val tmp    = Array.ofDim[Byte](size.noValues)
    var i      = 0
    while (i < result.length) {
      val noBytesRead = stream.read(tmp)
      assume(noBytesRead == size.noValues)
      val values = ArrayEx.map(
        tmp
      )(MathMacros.toUnsigned(_) * factor)
      result(i) = RealArrayTensor(size, values)
      i += 1
    }
    result
  }

  final private def loadImages(file: FileHandle)
  : Array[RealArrayTensor] = {
    using(
      new DataInputStream(
        new BufferedInputStream(
          file.openStream()
        )
      )
    )(loadImages)
  }

  final private def loadLabels(stream: DataInputStream): Array[Int] = {
    // Check magic number.
    val magic = stream.readInt
    assume(magic == 2049)

    // Get size of dataset.
    val noImages = stream.readInt
    assume(noImages > 0)

    // Read data and close stream.
    val data = new Array[Byte](noImages)
    assume(stream.read(data) == data.length)

    // Convert bytes to integers.
    ArrayEx.map(
      data
    )(_.toInt)
  }

  final private def loadLabels(file: FileHandle): Array[Int] = {
    using(
      new DataInputStream(
        new BufferedInputStream(
          file.openStream()
        )
      )
    )(loadLabels)
  }

  final private def load(imagesFile: FileHandle,
                         labelsFile: FileHandle)
  : Array[Batch] = {
    val images = loadImages(imagesFile)
    val labels = loadLabels(labelsFile)

    ArrayEx.zip(images, labels)((image, label) => Batch(
      image,
      SparseRealMatrixTensor(MatrixEx.labelsToSparse(10, label))
    ))
  }

  final def loadTrainingSet(file: FileHandle)
  : Array[Batch] = load(
    file ++ "train-images.idx3-ubyte",
    file ++ "train-labels.idx1-ubyte"
  )

  final def loadTestSet(file: FileHandle)
  : Array[Batch] = load(
    file ++ "t10k-images.idx3-ubyte",
    file ++ "t10k-labels.idx1-ubyte"
  )

}
