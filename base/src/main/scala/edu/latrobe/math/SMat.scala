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
object SMat {

  /*
  @inline
  final def apply(rows:     Int,
                  cols:     Int,
                  pointers: Array[Int],
                  indices:  Array[Int],
                  data:     Array[Real])
  : SMat = new CSCMatrix(data, rows, cols, pointers, indices)
  */


  //final val empty: SMat = zeros(0, 0)

  /*
  final def fromCols(noCols: Int, values: Iterable[(Int, SVec)]): SMat = {
    val noRows = if (values.isEmpty) 0 else values.head._2.length
    val result = zeros(noRows, noCols)
    val colIter = values.iterator
    while (colIter.hasNext) {
      val (c, col) = colIter.next()
      val rowIter = col.iterator
      while (rowIter.hasNext) {
        val (r, v) = rowIter.next()
        result.update(r, c, v)
      }
    }
    result
  }
  */


  /*
  final def fromRows(noRows: Int, values: Iterable[(Int, Transpose[SVec])])
  : SMat = {
    val noCols = if (values.isEmpty) 0 else values.head._2.t.length
    val result = zeros(noRows, noCols)
    val rowIter = values.iterator
    while (rowIter.hasNext) {
      val (r, row) = rowIter.next()
      val colIter = row.t.iterator
      while (colIter.hasNext) {
        val (c, v) = colIter.next()
        result.update(r, c, v)
      }
    }
    result
  }
  */


  // ---------------------------------------------------------------------------
  //  Label data creation.
  // ---------------------------------------------------------------------------


}
*/