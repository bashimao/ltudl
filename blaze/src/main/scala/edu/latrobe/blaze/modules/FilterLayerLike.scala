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
import scala.util.hashing._

trait FilterLayerLike[TBuilder <: FilterLayerLikeBuilder[_]]
  extends Layer[TBuilder]
    with TrainableLayer[TBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final val filterLayout
  : IndependentTensorLayout = builder.filterLayoutFor(inputLayoutHint)

  def filterReference
  : Option[LabeledBufferReference]

  def filter
  : ValueTensor

  @transient
  final override lazy val weightReferences
  : Set[LabeledBufferReference] = {
    val builder = Set.newBuilder[LabeledBufferReference]
    filterReference.map(builder += _)
    builder.result()
  }

  final override def extractWeightsFor(neuronNo: Long)
  : Array[Real] = {
    if (neuronNo >= 0L && neuronNo <= Int.MaxValue) {
      extractWeightsFor(neuronNo.toInt)
    }
    else {
      throw new IndexOutOfBoundsException
    }
  }

  def extractWeightsFor(neuronNo: Int)
  : Array[Real]


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveWeightGradients(input:     Tensor,
                                                       reference: Tensor,
                                                       output:    Tensor,
                                                       context:   PredictContext,
                                                       error:     Tensor,
                                                       sink:      ValueTensorBuffer)
  : Unit = {
    // Compute gradients depending on group selection.
    filterReference.foreach(fr => {
      val s = sink.get(fr)
      s.foreach(doDeriveFilterGradients(input, context, error, _))
    })
  }

  protected def doDeriveFilterGradients(input:   Tensor,
                                        context: PredictContext,
                                        error:   Tensor,
                                        sink:    ValueTensor)
  : Unit

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(input.layout, context, error)

  protected def doDeriveInputError(inputLayout: TensorLayout,
                                   context:     PredictContext,
                                   error:       Tensor)
  : Tensor

}

trait FilterLayerLikeBuilder[TThis <: FilterLayerLikeBuilder[_]]
  extends LayerBuilder[TThis]
    with TrainableLayerBuilder[TThis] {

  final private var _filterReference
  : LabeledBufferReference = LabeledBufferReference("filter")

  final def filterReference
  : LabeledBufferReference = _filterReference

  final def filterReference_=(value: LabeledBufferReference): Unit = {
    require(value != null)
    _filterReference = value
  }

  final def setFilterReference(value: LabeledBufferReference): TThis = {
    filterReference_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _filterReference :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _filterReference.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: FilterLayerLikeBuilder[TThis] =>
      _filterReference == other._filterReference
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: FilterLayerLikeBuilder[TThis] =>
        other._filterReference = _filterReference
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  def filterLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  final override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {
    filterReference_=(fn(_filterReference))
  }

}

/*
  final override def noNeurons: Long


  /**
   * @return multiplicative L1 & L2 weights regularization factor
   */
  //def lambda: (Real, Real)

  /**
   * To make L1 differentiable we do a pseudo approximation that requires
   * one additional parameter.
   */
  //def epsilon: Real

  //def noWeights: WBufferSize

  //def ticaLambda: Real

  //def ticaEpsilon: Real

}

*/
  /*
  final override def extractWeights: WeightsLike = Weights(
    if (wSource == null) DenseVector.vertcat(b, w) else b, DVec.empty
  )
  */

  /**
   * Approximates: sum(abs(wFlat))
   */
  //protected def computeRegularizerL1: Real = sqrt((wFlat o wFlat) + epsilon)

  //protected def computeRegularizerL2: Real = wFlat o wFlat

  //protected def computeRegularizerTICA(output: DSampleAct): Real

  //protected def computeRegularizerTICA(output: DBatchAct): Real

  /*
  final override def visualizeWeights(layout: (Int, Int), neuronDims: (Int, Int))
  : Bitmap = {
    // Select random neurons.
    var neurons = List[Long]()
    for (y <- 0 until layout._2; x <- 0 until layout._1) {
      neurons = Random.nextLong(noNeurons) :: neurons
    }
    visualizeWeights(layout, neuronDims, neurons)
  }
  */

  /*
  final override def computeCost(mode: ComputeMode, output: DSampleAct): Real = {
    var cost = super.computeCost(mode, output)
    /*
    if (!lambda._1.isNaN) {
      cost += lambda._1 * computeRegularizerL1
    }
    if (!lambda._2.isNaN) {
      cost += Real.zeroFive * lambda._2 * computeRegularizerL2
    }
    */
    val iter = weightPenalties.iterator
    while (iter.hasNext) {
      cost += iter.next().cost(mode, w, 1)
    }
    /*if (ticaLambda != Real.zero) {
      cost += ticaLambda * computeRegularizerTICA(output)
    }
    */
    cost
  }

  final override def computeCost(mode: ComputeMode, output: DBatchAct): Real = {
    var cost = super.computeCost(mode, output)
    /*
    if (!lambda._1.isNaN) {
      // Computes lasso regularizer for a set of weights. (Not differentiable!!!)
      // \lambda \sum_{i=0}^n \sum_{i=1}^k \abs{weights_{ij}}
      cost += lambda._1 * output.values.cols * computeRegularizerL1
    }
    if (!lambda._2.isNaN) {
      // Computes weight decay regularizer for a set of weights.
      // \lambda \sum_{i=0}^n \sum_{i=1}^k weights_{ij}^2
      cost += Real.zeroFive * lambda._2 * output.values.cols * computeRegularizerL2
    }
    */

    /*
    if (ticaLambda != Real.zero) {
      // TODO: Think about using mean to avoid excessive computation.
      cost += ticaLambda * /* output.values.cols * */ computeRegularizerTICA(output)
    }
    */
    cost
  }
*/

  /*
  def computeGradients(mode:          ComputeMode,
                       rawError:      DVec,
                       input:         SampleAct,
                       gradientsB:    DVec,
                       gradientsW:    DVec,
                       penaltyFactor: Int,
                       bufferNo:      Int)
  : Unit = {
    // Add regularizers.
    val iter = weightPenalties.iterator
    while (iter.hasNext) {
      iter.next().gradient(mode, w, penaltyFactor, gradientsW)
    }
  }

  def computeGradients(mode:          ComputeMode,
                       rawError:      DMat,
                       input:         BatchAct,
                       gradientsB:    DVec,
                       gradientsW:    DVec,
                       penaltyFactor: Int,
                       bufferNo:      Int)
  : Unit = {
    // Add regularizers.
    val iter = weightPenalties.iterator
    while (iter.hasNext) {
      iter.next().gradient(mode, w, penaltyFactor, gradientsW)
    }
  }
  */


/*
  override protected def computeRegularizerTICA(output: DSampleAct): Real = {
    val tmp = output.values
    Math.sqrt((tmp dot tmp) + ticaEpsilon) * outputSize
  }

  override protected def computeRegularizeTICA(output: DBatchAct)
  : Real = {
    val tmp = output.values.flatten()
    Math.sqrt((tmp dot tmp) + ticaEpsilon) * outputSize
  }
*/