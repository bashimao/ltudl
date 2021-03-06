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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._
import scala.util.hashing._

/**
  * f(x_a) = c
  *
  *       -1
  * f(x_a)   = impossible
  *
  * d f(x_a)
  * -------- = 0
  *  d x_a
  *
  */
abstract class SetValues
  extends NonTrainableMapLayerEx[SetValuesBuilder]
    with NonPenalizing {

  final protected val values
  : Array[Real] = builder.values.clone


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
  : Tensor = throw new UnsupportedOperationException

  final override protected def doPredictInvPerUnit(output:  Tensor,
                                                   context: PredictContext)
  : Tensor = throw new UnsupportedOperationException

  final override protected def doPredictInvPerChannel(output:  Tensor,
                                                      context: PredictContext)
  : Tensor = throw new UnsupportedOperationException

  final override protected def doPredictInvPerSample(output:  Tensor,
                                                     context: PredictContext)
  : Tensor = throw new UnsupportedOperationException

  final override protected def doPredictInvPerBatch(output:  Tensor,
                                                    context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


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
  : Tensor = {
    error.clear()
    error
  }

  final override protected def doDeriveInputErrorPerUnit(input:     Tensor,
                                                         reference: Tensor,
                                                         output:    Tensor,
                                                         context:   PredictContext,
                                                         error:     Tensor)
  : Tensor = {
    error.clear()
    error
  }

  final override protected def doDeriveInputErrorPerChannel(input:     Tensor,
                                                            reference: Tensor,
                                                            output:    Tensor,
                                                            context:   PredictContext,
                                                            error:     Tensor)
  : Tensor = {
    error.clear()
    error
  }

  final override protected def doDeriveInputErrorPerSample(input:     Tensor,
                                                           reference: Tensor,
                                                           output:    Tensor,
                                                           context:   PredictContext,
                                                           error:     Tensor)
  : Tensor = {
    error.clear()
    error
  }

  final override protected def doDeriveInputErrorPerBatch(input:     Tensor,
                                                          reference: Tensor,
                                                          output:    Tensor,
                                                          context:   PredictContext,
                                                          error:     Tensor)
  : Tensor = {
    error.clear()
    error
  }

}

final class SetValuesBuilder
  extends NonTrainableMapLayerExBuilder[SetValuesBuilder] {

  override def repr
  : SetValuesBuilder = this

  override def defaultDomain()
  : TensorDomain = TensorDomain.Batch

  private var _values
  : Array[Real] = new Array[Real](1)

  def values
  : Array[Real] = _values

  def values_=(value: Array[Real])
  : Unit = {
    require(value != null)
    _values = value
  }

  def setValues(value: Array[Real])
  : SetValuesBuilder = {
    values_=(value)
    this
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
  : Boolean = that.isInstanceOf[SetValuesBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SetValuesBuilder =>
      ArrayEx.compare(_values, other._values)
    case _ =>
      false
  })

  override protected def doCopy()
  : SetValuesBuilder = SetValuesBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SetValuesBuilder =>
        other._values = _values.clone
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = SetValuesBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SetValuesBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

}

object SetValuesBuilder
  extends ModuleVariantTable[SetValuesBuilder] {

  register(2, SetValues_JVM_Baseline_Description)

  final def apply()
  : SetValuesBuilder = new SetValuesBuilder

  final def apply(domain: TensorDomain)
  : SetValuesBuilder = apply().setDomain(domain)

  final def apply(domain: TensorDomain, values: Array[Real])
  : SetValuesBuilder = apply(domain).setValues(values)

  final def apply(domain: TensorDomain, value: Real)
  : SetValuesBuilder = apply(domain, Array(value))

}
