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

package edu.latrobe.math

import breeze.linalg._
import edu.latrobe._
import scala.collection._

/*
object SVec {

  /*
  @inline
  final def derive(length:  Int,
                   indices: TraversableOnce[Int],
                   values:  TraversableOnce[Real])
  : SVec = apply(length, indices.toArray, values.toArray)
  */

  /*
  final def tabulate(length: Int)(fn: Int => Real): SVec = {
    val result = zeros(length)
    var i = 0
    while (i < length) {
      result.update(i, fn(i))
      i += 1
    }
    result
  }
  */
/*
  @inline
  final def derive(value0: Real)
  : DenseMatrix[Real] = DenseMatrix.create(1, 1, Array(value0))

  @inline
  final def derive(rows: Int,value0: Real, valuesN: Real*)
  : DenseMatrix[Real] = derive(rows, value0 :: valuesN.toList)

  @inline
  final def derive[T](rows: Int, values: TraversableOnce[T])
  : DenseMatrix[T] = {
    val data = values.toArray
    val cols = data.length / rows
    require(data.length == rows * cols)
    DenseMatrix.create(rows, cols, data)
  }

  final def empty[T]: DenseMatrix[T] = DenseMatrix.zeros[T](0, 0)
  */

}
*/