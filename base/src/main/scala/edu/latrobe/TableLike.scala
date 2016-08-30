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

package edu.latrobe

import scala.reflect._

/**
  * All tables must allow backward indexing using negative indices.
  */
trait TableLike[T] {

  def length
  : Int

  def getEntry(index: Int
              ): T

  final def getEntries(indices: Range)
                      (implicit tagT: ClassTag[T])
  : Array[T] = RangeEx.map(indices)(getEntry)

  def iterator
  : Iterator[T]

}
