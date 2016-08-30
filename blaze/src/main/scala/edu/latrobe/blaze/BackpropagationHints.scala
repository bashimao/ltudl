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

final case class BackpropagationHints(inputDependency:  TensorDependency,
                                      outputDependency: TensorDependency)
  extends Serializable {
}

object BackpropagationHints {

  final val requireAny: BackpropagationHints = BackpropagationHints(
    TensorDependency.RequiresEither,
    TensorDependency.RequiresEither
  )

  final val requireNeither: BackpropagationHints = BackpropagationHints(
    TensorDependency.NotRequired,
    TensorDependency.NotRequired
  )

  final val requireInput: BackpropagationHints = BackpropagationHints(
    TensorDependency.Required,
    TensorDependency.NotRequired
  )

  final val requireOutput: BackpropagationHints = BackpropagationHints(
    TensorDependency.NotRequired,
    TensorDependency.Required
  )

  final val requireBoth: BackpropagationHints = BackpropagationHints(
    TensorDependency.Required,
    TensorDependency.Required
  )

}