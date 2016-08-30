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

package edu.latrobe.blaze.regularizers

import edu.latrobe.blaze._

/**
  * Dependent regularizers control the execution of child regularizers.
  */
abstract class ComplexRegularizer[TBuilder <: ComplexRegularizerBuilder[_]]
  extends RegularizerEx[TBuilder] {

  final val children
  : Seq[Regularizer] = builder.children.map(
    _.build(platformHint, seed)
  )

}

abstract class ComplexRegularizerBuilder[TThis <: ComplexRegularizerBuilder[_]]
  extends RegularizerExBuilder[TThis] {

  def children
  : Seq[RegularizerBuilder]

}
