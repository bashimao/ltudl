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

import edu.latrobe._
import scala.collection._

/**
 * This is a primitive that represents the input error of the current layer.
 * It is only evaluated if there is a next layer that needs it. The memory
 * associated with the previous error can be freed automatically.
 *
 * This structure is most useful for the lazy daisy chaining pattern we apply
 * during backprop and CAN save a lot of memory there.
 *
 * @param sources Either a deallocatable tensors or null.
 * @param transformFn The function to execute if get is called.
 */
// TODO: Lazy is a term that does not really cover the daisy chaining concept here. Should find a better name!
abstract class NextError
  extends AutoCloseable {

  final def compute(): Tensor = {
    val result = doCompute()
    close()
    result
  }

  protected def doCompute(): Tensor

}
