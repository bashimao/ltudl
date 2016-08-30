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

final class DependentError(private val parent:      NextError,
                           private val doComputeFn: Tensor => Tensor)
  extends NextError {
  require(parent != null && doComputeFn != null)

  override def close()
  : Unit = parent.close()

  override protected def doCompute()
  : Tensor = {
    val source = parent.compute()
    val result = doComputeFn(source)
    if (result ne source) {
      source.close()
    }
    result
  }

}

object DependentError {

  final def apply(parent: NextError, doComputeFn: Tensor => Tensor)
  : DependentError = new DependentError(parent, doComputeFn)

}

final class DependentErrorEx(private val parents:     Array[NextError],
                             private val doComputeFn: Array[Tensor] => Tensor)
  extends NextError {
  require(
    parents != null && !ArrayEx.contains(parents, null) && doComputeFn != null
  )

  override def close()
  : Unit = {
    ArrayEx.foreach(
      parents
    )(_.close())
  }

  override protected def doCompute()
  : Tensor = {
    val values = ArrayEx.map(
      parents
    )(_.compute())
    val result = doComputeFn(values)
    ArrayEx.foreach(values)(values => {
      if (values ne result) {
        values.close()
      }
    })
    result
  }

}

object DependentErrorEx {

  final def apply(parents:     Array[NextError],
                  doComputeFn: Array[Tensor] => Tensor)
  : DependentErrorEx = new DependentErrorEx(parents, doComputeFn)

}
