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
import scala.collection._

abstract class GradientCheck
  extends InstanceEx[GradientCheckBuilder] {

  final def apply(phaseNo:   Long,
                  model:     Module,
                  input:     Tensor,
                  reference: Tensor)
  : GradientDeviation = apply(
    phaseNo,
    model,
    Traversable.empty,
    input,
    reference
  )

  final def apply(phaseNo: Long,
                  model:   Module,
                  batch:   Batch)
  : GradientDeviation = apply(
    phaseNo,
    model,
    Traversable.empty,
    batch
  )

  final def apply(phaseNo: Long,
                  model:   Module,
                  batches: Traversable[Batch])
  : GradientDeviation = apply(
    phaseNo,
    model,
    Traversable.empty,
    batches
  )

  final def apply(phaseNo:      Long,
                  model:        Module,
                  batches:      Array[Batch])
  : GradientDeviation = apply(
    phaseNo,
    model,
    Traversable.empty,
    batches
  )

  final def apply(phaseNo:      Long,
                  model:        Module,
                  regularizers: Traversable[Regularizer],
                  input:        Tensor,
                  reference:    Tensor)
  : GradientDeviation = {
    val mode = Training.reproducible(phaseNo)
    using(model.weightBuffer.allocateZeroedSibling())(gradients => {
      val baseCost = {
        model.refresh()
        val context = model.predict(mode, input, reference)
        regularizers.foreach(
          context.value += _.evaluate(phaseNo, model.weightBuffer, context)
        )
        model.deriveGradients(context, gradients)
        regularizers.foreach(
          _.deriveGradients(phaseNo, model.weightBuffer, context, gradients)
        )
        val value = context.value
        context.close()
        value
      }

      val result = doApply(model.weightBuffer, baseCost, gradients, () => {
        model.refresh()
        val context = model.predict(mode, input, reference)
        regularizers.foreach(
          context.value += _.evaluate(phaseNo, model.weightBuffer, context)
        )
        val value = context.value
        context.close()
        value
      })

      // Make sure the model is refreshed, in case we want to do anything else with it.
      result
    })
  }

  final def apply(phaseNo:      Long,
                  model:        Module,
                  regularizers: Traversable[Regularizer],
                  batch:        Batch)
  : GradientDeviation = {
    apply(
      phaseNo,
      model,
      regularizers,
      batch.input,
      batch.output
    )
  }

  final def apply(phaseNo:      Long,
                  model:        Module,
                  regularizers: Traversable[Regularizer],
                  batches:      Traversable[Batch])
  : GradientDeviation = {
    batches.foldLeft(
      GradientDeviation.zero
    )(_ + apply(phaseNo, model, regularizers, _))
  }

  final def apply(phaseNo:      Long,
                  model:        Module,
                  regularizers: Traversable[Regularizer],
                  batches:      Array[Batch])
  : GradientDeviation = {
    ArrayEx.foldLeft(
      GradientDeviation.zero,
      batches
    )(_ + apply(phaseNo, model, regularizers, _))
  }

  protected def doApply(weights:   ValueTensorBuffer,
                        baseCost:  Real,
                        gradients: ValueTensorBuffer,
                        fnCost:    () => Real)
  : GradientDeviation

}

abstract class GradientCheckBuilder
  extends InstanceExBuilder0[GradientCheckBuilder, GradientCheck] {
}

abstract class GradientCheckEx[TBuilder <: GradientCheckExBuilder[_]]
  extends GradientCheck {

  override def builder
  : TBuilder

}

abstract class GradientCheckExBuilder[TThis <: GradientCheckExBuilder[_]]
  extends GradientCheckBuilder {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

}