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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.batchpools._
import edu.latrobe.time._
import java.io.OutputStream
import scala.collection._
import scala.util.hashing._

final class CrossValidation(override val builder: CrossValidationBuilder,
                            override val seed:    InstanceSeed)
  extends DependentObjectiveEx[CrossValidationBuilder] {
  require(builder != null && seed != null)

  val batchPool
  : BatchPool = builder.batchPool.build(
    builder.inputLayoutHint,
    builder.batches,
    seed
  )

  val noBatches
  : Int = builder.noBatches

  override protected def doClose()
  : Unit = {
    batchPool.close()
    super.doClose()
  }

  override protected def doEvaluate(sink:                Sink,
                                    optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    var i = 0
    while (i < noBatches) {
      using(batchPool.draw())(drawContext => {
        val batch = drawContext.batch

        using(
          model.predict(Inference(), batch).dropIntermediates()
        )(prediction => {
          // Call children.
          val result = super.doEvaluate(
            sink,
            optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
            model,
            batch, prediction.output, prediction.value
          )
          if (result.isDefined) {
            return result
          }
        })
      })

      i += 1
    }
    None
  }

}

final class CrossValidationBuilder
  extends DependentObjectiveExBuilder[CrossValidationBuilder] {

  override def repr
  : CrossValidationBuilder = this

  private var _inputLayoutHint
  : TensorLayout = IndependentTensorLayout.zero

  def inputLayoutHint
  : TensorLayout = _inputLayoutHint

  def inputLayoutHint_=(value: TensorLayout)
  : Unit = {
    require(value != null)
    _inputLayoutHint = value
  }

  def setInputLayoutHint(value: TensorLayout)
  : CrossValidationBuilder = {
    inputLayoutHint_=(value)
    this
  }

  private var _batches
  : Iterable[Batch] = Iterable.empty

  def batches
  : Iterable[Batch] = _batches

  def batches_=(value: Iterable[Batch])
  : Unit = {
    require(!value.exists(_ == null))
    _batches = value
  }

  def setBatches(value: Iterable[Batch])
  : CrossValidationBuilder = {
    batches_=(value)
    this
  }

  private var _batchPool
  : BatchPoolBuilder = ChooseAtRandomBuilder()

  def batchPool
  : BatchPoolBuilder = _batchPool

  def batchPool_=(value: BatchPoolBuilder)
  : Unit = {
    require(value != null)
    _batchPool = value
  }

  def setBatchPool(value: BatchPoolBuilder)
  : CrossValidationBuilder = {
    batchPool_=(value)
    this
  }

  private var _noBatches
  : Int = 1

  def noBatches
  : Int = _noBatches

  def noBatches_=(value: Int)
  : Unit = {
    require(value > 0)
    _noBatches = value
  }

  def setNoBatches(value: Int)
  : CrossValidationBuilder = {
    noBatches_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    s"${_inputLayoutHint} x ${_batches.size} => ${_batchPool}" :: _noBatches :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _inputLayoutHint.hashCode())
    tmp = MurmurHash3.mix(tmp, _batches.hashCode())
    tmp = MurmurHash3.mix(tmp, _batchPool.hashCode())
    tmp = MurmurHash3.mix(tmp, _noBatches.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CrossValidationBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CrossValidationBuilder =>
      _inputLayoutHint == other._inputLayoutHint &&
      _batches         == other._batches         &&
      _batchPool       == other._batchPool       &&
      _noBatches       == other._noBatches
    case _ =>
      false
  })

  override protected def doCopy()
  : CrossValidationBuilder = CrossValidationBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CrossValidationBuilder =>
        other._inputLayoutHint = _inputLayoutHint
        other._batches         = _batches
        other._batchPool       = _batchPool.copy
        other._noBatches       = _noBatches
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : CrossValidation = new CrossValidation(this, seed)

}

object CrossValidationBuilder {

  final def apply()
  : CrossValidationBuilder = new CrossValidationBuilder

  final def apply(inputLayoutHint: TensorLayout,
                  batches:         Iterable[Batch],
                  batchPool:       BatchPoolBuilder,
                  noBatches:       Int)
  : CrossValidationBuilder = apply(
  ).setInputLayoutHint(
    inputLayoutHint
  ).setBatches(
    batches
  ).setBatchPool(
    batchPool
  ).setNoBatches(noBatches)

  final def apply(inputLayoutHint: TensorLayout,
                  batches:         Iterable[Batch],
                  batchPool:       BatchPoolBuilder,
                  noBatches:       Int,
                  child0:          ObjectiveBuilder)
  : CrossValidationBuilder = apply(
    inputLayoutHint, batches, batchPool, noBatches
  ) += child0

  final def apply(inputLayoutHint: TensorLayout,
                  batches:         Iterable[Batch],
                  batchPool:       BatchPoolBuilder,
                  noBatches:       Int,
                  child0:          ObjectiveBuilder,
                  childN:          ObjectiveBuilder*)
  : CrossValidationBuilder = apply(
    inputLayoutHint, batches, batchPool, noBatches, child0
  ) ++= childN

  final def apply(inputLayoutHint: TensorLayout,
                  batches:         Iterable[Batch],
                  batchPool:       BatchPoolBuilder,
                  noBatches:       Int,
                  childN:          TraversableOnce[ObjectiveBuilder])
  : CrossValidationBuilder = apply(
    inputLayoutHint, batches, batchPool, noBatches
  ) ++= childN

}

final case class CrossValidationState(override val parent: InstanceState,
                                      batchPool:           InstanceState)
  extends InstanceState {
}
