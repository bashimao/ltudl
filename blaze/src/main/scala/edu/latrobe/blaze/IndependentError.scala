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

package edu.latrobe.blaze

import edu.latrobe._

final class IndependentError(private val source:      Tensor,
                             private val doComputeFn: Tensor => Tensor)
  extends NextError {
  require(doComputeFn != null)

  override def toString
  : String = s"IndependentError[$source, $doComputeFn]"

  private var handedOffValues
  : Boolean = false

  override def close()
  : Unit = {
    if (!handedOffValues) {
      if (source != null) {
        source.tryClose()
      }
    }
  }

  override protected def doCompute()
  : Tensor = {
    val result = doComputeFn(source)
    if (result ne source) {
      if (source != null) {
        source.close()
      }
    }
    else {
      handedOffValues = true
    }
    result
  }

}

object IndependentError {

  final def apply(doComputeFn: => Tensor)
  : IndependentError = apply(null, x => doComputeFn)

  final def apply(values: Tensor, doComputeFn: Tensor => Tensor)
  : IndependentError = new IndependentError(values, doComputeFn)

  final def derive(values: Tensor)
  : IndependentError = apply(values, values => values)

}
