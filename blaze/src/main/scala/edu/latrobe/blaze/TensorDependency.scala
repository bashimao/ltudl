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

abstract class TensorDependency
  extends Serializable

object TensorDependency {

  case object NotRequired
    extends TensorDependency

  case object Required
    extends TensorDependency

  /**
    * If input and output are interchangeable, blaze will take the input if it
    * it has been pinned in memory. Otherwise, it will lean towards the output
    * since that one has the potential to become useful for the next layer.
    *
    * This - of course implies that both Interchangeable cannot be mixed with
    * Required or NotRequired.
    */
  case object RequiresEither
    extends TensorDependency

}
