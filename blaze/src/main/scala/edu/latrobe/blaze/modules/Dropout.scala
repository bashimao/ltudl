/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.generic._
import scala.util.hashing._

/**
  * Inverse dropout implementation.
  * (Can also do original dropout, but nobody really uses that anymore!)
  *
  * Inverse dropout boosts dropout rate during training, and has
  * zero cost during inference.
  *
  * Randomly drops out values. Works best with PseudoRNG.default.uniform or
  * PseudoRNG(x).uniform.
  *
  * This is equivalent to setting the v2 flag in Torch.
  *
  *
  * Let's presume we dropout 33%. We generate a tensor of the same size and fill
  * it with random ones and zeros:
  *
  * m_a = rand[0 or 1]
  *
  * f(x_a, m_a) = x_a * m_a
  *
  * Hence:
  *   f(x_a | m_a = 0) = 0
  *   f(x_a | m_a = 1) = x_a
  *
  * d f(x_a, m_a)
  * ------------- = m_a
  *     d x_a
  *
  * d f(x_a, m_a)
  * ------------- = 0
  *  d x_b, b!=a
  *
  *                 ---
  * D f(x_a, m_a)   \   d f(x_a, m_a)
  * ------------- = /   ------------- di = m_a di
  *    D x_a        ---      x_i
  *                  i
  *
  *
  * However, this is only for the training phase. In the inference face we will
  * not dropout. So, presuming we dropped 50% out during training, the magnitude
  * of the activation vector is twice (1/0.5 = 2) as large. If we dropped out
  * 67%, the values magnitude will be about 3 times larger (1 / (1 - 0.67) ~= 3).
  *
  * The original dropout formula compensated for this by scaling down the
  * activations in inference mode to shift the magnitude on average into the
  * same domain as it was during training.
  * So if:
  *
  * p = 90% during training, multiply 1 - 0.90 during inference.
  * p = 50% during training, multiply 1 - 0.50 during inference.
  * p = 67% during training, multiply 1 - 0.67 during inference.
  * p = 10% during training, multiply 1 - 0.10 during inference.
  * p =  0% during training, multiply 1 - 0.00 during inference.
  *
  * Obviously, this is not desirable, because the network's complexity is
  * increased slightly and inference becomes slower.
  *
  * So instead muliplying (1 - p) during inference, almost everybody boosts the
  * mask values so that the activation's magnitude on average is in the same
  * domain as if we would have not dropped out any inputs.
  * So if:
  *
  * p = 90% during training, multiply mask with (1 / (1 - 0.90) ~= 10).
  * p = 67% during training, multiply mask with (1 / (1 - 0.67) ~= 3).
  * p = 50% during training, multiply mask with (1 / (1 - 0.50) ~= 2).
  * p = 10% during training, multiply mask with (1 / (1 - 0.10) ~= 1.1).
  * p =  0% during training, multiply mask with (1 / (1 - 0.00) ~= 1).
  *
  * In this case no action has to be taken during backprop.
  *
  */
abstract class Dropout
  extends NonTrainableMapLayer[DropoutBuilder]
    with NonPenalizing {

  final val useOriginalAlgorithm
  : Boolean = builder.useOriginalAlgorithm

  final val probability
  : Real = builder.probability

  final val probabilityInv
  : Real = Real.one - probability

  final val boostFactor
  : Real = Real.one / probabilityInv


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = mode match {
    case mode: Training =>
      // Keep parameters stable during gradient testing.
      val rng = {
        if (mode.reproducible) {
          PseudoRNG(System.identityHashCode(this))
        }
        else {
          this.rng
        }
      }
      doPredictForTraining(inPlaceAllowed, input, rng)

    case mode: Inference =>
      val out = doPredictForInference(inPlaceAllowed, input)
      (out, EmptyContext)

    case _ =>
      throw new MatchError(mode)
  }

  protected def doPredictForTraining(inPlaceAllowed: Boolean,
                                     input:          Tensor,
                                     rng:            PseudoRNG)
  : (Tensor, PredictContext)

  protected def doPredictForInference(inPlaceAllowed: Boolean,
                                      input:          Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(context, error)

  protected def doDeriveInputError(context: PredictContext,
                                   error:   Tensor)
  : Tensor

}

final class DropoutBuilder
  extends NonTrainableMapLayerBuilder[DropoutBuilder] {

  override def repr
  : DropoutBuilder = this

  /**
    * Probability of each unit to drop out.
    */
  private var _probability
  : Real = Real.pointFive

  def probability
  : Real = _probability

  def probability_=(value: Real)
  : Unit = {
    require(value >= Real.zero && value < 0.99999f)
    _probability = value
  }

  def setProbability(value: Real): DropoutBuilder = {
    probability_=(value)
    repr
  }

  /**
    * If this is set to true the original dropout algorithm will be used
    * instead of the default that everybody uses nowadays  and which is
    * sometimes also called inverted dropout.
    */
  var useOriginalAlgorithm
  : Boolean = false

  def setUseOriginalAlgorithm(value: Boolean)
  : DropoutBuilder = {
    useOriginalAlgorithm_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    f"${_probability}%.4g" :: useOriginalAlgorithm :: super.doToString()
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DropoutBuilder]

  override protected def doCopy()
  : DropoutBuilder = DropoutBuilder()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _probability.hashCode())
    tmp = MurmurHash3.mix(tmp, useOriginalAlgorithm.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DropoutBuilder =>
      _probability         == other._probability &&
      useOriginalAlgorithm == other.useOriginalAlgorithm
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DropoutBuilder =>
        other._probability         = _probability
        other.useOriginalAlgorithm = useOriginalAlgorithm
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = DropoutBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = DropoutBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object DropoutBuilder
  extends ModuleVariantTable[DropoutBuilder] {

  register(64, Dropout_Generic_Baseline_Description)

  final def apply()
  : DropoutBuilder = new DropoutBuilder

  final def apply(probability: Real)
  : DropoutBuilder = apply().setProbability(probability)
}
