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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._
import scala.util.hashing._

/**
  * f(x_a) = x_a * c
  *
  *      -1
  * f(x_a)  = x / c
  *
  * f(x_a)
  * ------ = c
  *  x_a
  */
abstract class MultiplyValues
  extends NonTrainableMapLayerEx[MultiplyValuesBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerValue(mode:           Mode,
                                                 inPlaceAllowed: Boolean,
                                                 input:          Tensor,
                                                 reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerValue(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerUnit(mode:           Mode,
                                                inPlaceAllowed: Boolean,
                                                input:          Tensor,
                                                reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerUnit(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerChannel(mode:           Mode,
                                                   inPlaceAllowed: Boolean,
                                                   input:          Tensor,
                                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerChannel(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerSample(mode:           Mode,
                                                  inPlaceAllowed: Boolean,
                                                  input:          Tensor,
                                                  reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerSample(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  final override protected def doPredictPerBatch(mode:           Mode,
                                                 inPlaceAllowed: Boolean,
                                                 input:          Tensor,
                                                 reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredictPerBatch(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  protected def doPredictPerValue(inPlaceAllowed: Boolean,
                                  input:          Tensor)
  : Tensor

  protected def doPredictPerUnit(inPlaceAllowed: Boolean,
                                 input:          Tensor)
  : Tensor

  protected def doPredictPerChannel(inPlaceAllowed: Boolean,
                                    input:          Tensor)
  : Tensor

  protected def doPredictPerSample(inPlaceAllowed: Boolean,
                                   input:          Tensor)
  : Tensor

  protected def doPredictPerBatch(inPlaceAllowed: Boolean,
                                  input:          Tensor)
  : Tensor

  final override protected def doPredictInvPerValue(output:  Tensor,
                                                    context: PredictContext)
  : Tensor = doPredictInvPerValue(output)

  final override protected def doPredictInvPerUnit(output:  Tensor,
                                                   context: PredictContext)
  : Tensor = doPredictInvPerUnit(output)

  final override protected def doPredictInvPerChannel(output:  Tensor,
                                                      context: PredictContext)
  : Tensor = doPredictInvPerChannel(output)

  final override protected def doPredictInvPerSample(output:  Tensor,
                                                     context: PredictContext)
  : Tensor = doPredictInvPerSample(output)

  final override protected def doPredictInvPerBatch(output:  Tensor,
                                                    context: PredictContext)
  : Tensor = doPredictInvPerBatch(output)

  protected def doPredictInvPerValue(output: Tensor)
  : Tensor

  protected def doPredictInvPerUnit(output: Tensor)
  : Tensor

  protected def doPredictInvPerChannel(output: Tensor)
  : Tensor

  protected def doPredictInvPerSample(output: Tensor)
  : Tensor

  protected def doPredictInvPerBatch(output: Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputErrorPerValue(input:     Tensor,
                                                          reference: Tensor,
                                                          output:    Tensor,
                                                          context:   PredictContext,
                                                          error:     Tensor)
  : Tensor = doDeriveInputErrorPerValue(error)

  final override protected def doDeriveInputErrorPerUnit(input:     Tensor,
                                                         reference: Tensor,
                                                         output:    Tensor,
                                                         context:   PredictContext,
                                                         error:     Tensor)
  : Tensor = doDeriveInputErrorPerUnit(error)

  final override protected def doDeriveInputErrorPerChannel(input:     Tensor,
                                                            reference: Tensor,
                                                            output:    Tensor,
                                                            context:   PredictContext,
                                                            error:     Tensor)
  : Tensor = doDeriveInputErrorPerChannel(error)

  final override protected def doDeriveInputErrorPerSample(input:     Tensor,
                                                           reference: Tensor,
                                                           output:    Tensor,
                                                           context:   PredictContext,
                                                           error:     Tensor)
  : Tensor = doDeriveInputErrorPerSample(error)

  final override protected def doDeriveInputErrorPerBatch(input:     Tensor,
                                                          reference: Tensor,
                                                          output:    Tensor,
                                                          context:   PredictContext,
                                                          error:     Tensor)
  : Tensor = doDeriveInputErrorPerBatch(error)

  protected def doDeriveInputErrorPerValue(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerUnit(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerChannel(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerSample(error: Tensor)
  : Tensor

  protected def doDeriveInputErrorPerBatch(error: Tensor)
  : Tensor

}

final class MultiplyValuesBuilder
  extends NonTrainableMapLayerExBuilder[MultiplyValuesBuilder] {

  override def repr
  : MultiplyValuesBuilder = this

  override def defaultDomain()
  : TensorDomain = TensorDomain.Batch

  private var _values
  : Array[Real] = ArrayEx.fill(1)(Real.one)

  def values
  : Array[Real] = _values

  def values_=(value: Array[Real])
  : Unit = {
    require(value.length > 0)
    _values = value
  }

  def setValues(value: Array[Real])
  : MultiplyValuesBuilder = {
    values_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = {
    val builder = List.newBuilder[String]
    if (_values.length <= 4) {
      var i = 0
      while (i < _values.length) {
        builder += f"${_values(i)}%.4g"
        i += 1
      }
    }
    else {
      builder += s"${_values.length} values"
    }
    builder.result() ::: super.doToString()
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), ArrayEx.hashCode(_values))

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MultiplyValuesBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MultiplyValuesBuilder =>
      ArrayEx.compare(_values, other._values)
    case _ =>
      false
  })

  override protected def doCopy()
  : MultiplyValuesBuilder = MultiplyValuesBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: MultiplyValuesBuilder =>
        other._values = _values.clone
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = MultiplyValuesBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = MultiplyValuesBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

}

object MultiplyValuesBuilder
  extends ModuleVariantTable[MultiplyValuesBuilder] {

  register(2, MultiplyValues_JVM_Baseline_Description)

  final def apply()
  : MultiplyValuesBuilder = new MultiplyValuesBuilder

  final def apply(domain: TensorDomain)
  : MultiplyValuesBuilder = apply().setDomain(domain)

  final def apply(domain: TensorDomain, values: Array[Real])
  : MultiplyValuesBuilder = apply(domain).setValues(values)

  final def apply(domain: TensorDomain, value: Real)
  : MultiplyValuesBuilder = apply(domain, Array(value))

}
