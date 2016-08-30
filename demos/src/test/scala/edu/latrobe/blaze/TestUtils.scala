/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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

package edu.latrobe.blaze

import breeze.linalg.{DenseMatrix, DenseVector}
import edu.latrobe._

object TestUtils {

  val tolerance0: Real = 1e-5f

  val tolerance1: Real = 1e-4f

  val tolerance2: Real = 1e-3f

  final def similarity(a: DenseVector[Real], b: DenseVector[Real]): Real = {
    val diff = a - b
    val result = VectorEx.dot(diff, diff) / diff.size
    println(f"Similarity: $result%.4g")
    result
  }

  final def similarity(a: DenseMatrix[Real], b: DenseMatrix[Real]): Real = {
    val diff = a - b
    val result = MatrixEx.dot(diff, diff) / diff.size
    println(f"Similarity: $result%.4g")
    if (Real.isNaN(result)) {
      throw new UnknownError
    }
    result
  }

  final def similarity(a: Tensor, b: Tensor)
  : Real = similarity(a.valuesMatrix, b.valuesMatrix)

}
