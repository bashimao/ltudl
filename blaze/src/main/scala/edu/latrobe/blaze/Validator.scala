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

abstract class Validator
  extends InstanceEx[ValidatorBuilder] {

  def apply(reference: Tensor, output: Tensor)
  : ValidationScore

  /*
  final def apply(reference: TensorSet, output: TensorSet)
  : ValidationScore = reference.tensors.fastFoldLeftEx(
    ValidationScore.zero, output.tensors
  )(_ + apply(_, _))

  final def apply(reference: TensorLike, output: TensorLike)
  : ValidationScore = (reference, output) match {
    case (reference: Tensor,    output: Tensor   ) => apply(reference, output)
    case (reference: TensorSet, output: TensorSet) => apply(reference, output)
  }
  */

}

abstract class ValidatorBuilder
  extends InstanceExBuilder0[ValidatorBuilder, Validator] {
}

abstract class ValidatorEx[TBuilder <: ValidatorExBuilder[_]]
  extends Validator {

  override def builder
  : TBuilder

}

abstract class ValidatorExBuilder[TThis <: ValidatorExBuilder[_]]
  extends ValidatorBuilder {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

}