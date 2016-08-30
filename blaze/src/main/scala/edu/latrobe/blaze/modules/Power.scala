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
import edu.latrobe.blaze.TensorDependency._
import scala.util.hashing._

/**
 *               p
 * Predict: f = x
 *
 *           -1    p  /---
 * Inverse: f   = -  /  y
 *                 \/
 *
 *           d f      p - 1
 * Gradient: --- = p x
 *           d x
 *
 */
abstract class Power
  extends NonTrainableMapLayer[PowerBuilder]
    with NonPenalizing {

  final val exponent
  : Real = builder.exponent


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredict(input)
    (out, EmptyContext)
  }

  protected def doPredict(input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = doPredictInv(output)

  protected def doPredictInv(output: Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(input, error)

  protected def doDeriveInputError(input: Tensor,
                                   error: Tensor)
  : Tensor

}

final class PowerBuilder
  extends NonTrainableMapLayerBuilder[PowerBuilder] {

  override def repr
  : PowerBuilder = this

  var exponent
  : Real = Real.two
  
  def setExponent(value: Real)
  : PowerBuilder = {
    exponent_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$exponent%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PowerBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), exponent.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PowerBuilder =>
      exponent == other.exponent
    case _ =>
      false
  })

  override protected def doCopy()
  : PowerBuilder = PowerBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PowerBuilder =>
        other.exponent = exponent
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = PowerBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = PowerBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object PowerBuilder extends ModuleVariantTable[PowerBuilder] {

  register(2, Power_JVM_Baseline_Description)

  final def apply()
  : PowerBuilder = new PowerBuilder

  final def apply(exponent: Real)
  : PowerBuilder = apply().setExponent(exponent)

}