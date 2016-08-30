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

package edu.latrobe

import scala.collection._

package object blaze {

  /**
    * Time in seconds under which benchmark results will not be written to stdout.
    */
  final val BLAZE_BENCHMARK_REPORTING_THRESHOLD
  : Real = Environment.parseReal(
    "BLAZE_BENCHMARK_REPORTING_THRESHOLD",
    Real.zero,
    x => x >= Real.zero
  )

  /*
  final val BLAZE_PARALLELIZATION_THRESHOLD_MIN: Int = {
    env.getOrElse("BLAZE_PARALLELIZATION_THRESHOLD_MIN", (16 * 1024).toString).toInt
  }
  assume(BLAZE_PARALLELIZATION_THRESHOLD_MIN > 0)
  logger.info(s"BLAZE_PARALLELIZATION_THRESHOLD_MIN=$BLAZE_PARALLELIZATION_THRESHOLD_MIN")

  final val BLAZE_PARALLELIZATION_THRESHOLD_MAX: Int = {
    env.getOrElse("BLAZE_PARALLELIZATION_THRESHOLD_MAX", (128 * 1024).toString).toInt
  }
  assume(BLAZE_PARALLELIZATION_THRESHOLD_MAX > BLAZE_PARALLELIZATION_THRESHOLD_MIN)
  logger.info(s"BLAZE_PARALLELIZATION_THRESHOLD_MAX=$BLAZE_PARALLELIZATION_THRESHOLD_MAX")
  */

  //type ID = Long

  /*
  final implicit class RealFunctions[T](r: Real) {

    @inline
    def ++(other: DVec): DVec = {
      val result = DVec.zeros(1 + other.length)
      result.unsafeUpdate(0, r)
      result(1 until result.length) := other
      result
    }

    @inline
    def :++(other: DVec): DVec = {
      val result = DVec.zeros(1 + other.length)
      result.unsafeUpdate(0, r)
      result(1 until result.length) := other
      result
    }

  }
  */

  /*
  @inline
  final def transformExDirect[T, U, V](dv: DenseVector[T], other: DenseVector[U], v3: V)(fn: (T, U, V) => T): Unit = {
    require(dv.length == other.length)

    val length = dv.length

    val data1   = other.data
    val stride1 = other.stride
    val data0   = dv.data
    val stride0 = dv.stride

    if (stride0 == 1 && stride1 == 1) {
      val offset1 = other.offset
      val offset0 = dv.offset
      cforRange(0 until length)(i => {
        val tmp = offset0 + i
        data0(tmp) = fn(data0(tmp), data1(offset1 + i), v3)
      })
    }
    else {
      // TODO: Java range check can be eliminated in some cases!
      var offset1 = other.offset
      var offset0 = dv.offset
      val end0    = dv.endOffset
      while (offset0 < end0) {
        data0(offset0) = fn(data0(offset0), data1(offset1), v3)
        offset1 += stride1
        offset0 += stride0
      }
    }
  }*/

  /*
  final implicit class DenseVectorFunctions[T](dv: DenseVector[T]) {

    /*
        @inline
        def demultiplex(sequenceLength: Int)
                       (implicit classTag: ClassTag[T], zero: Zero[T])
        : DenseVector[T] = {
          val result = DenseVector.zeros[T](dv.length)
          demultiplex(sequenceLength, result)
          result
        }

        @inline
        def demultiplex(sequenceLength: Int, result: DenseVector[T]): Unit = {
          require(dv.length == result.length)

          val noSequences = dv.length / sequenceLength
          assume(noSequences * sequenceLength == dv.length)

          // Fetch frequently used values.
          val data1      = result.data
          val valStride1 = result.stride
          val seqStride1 = result.stride * noSequences
          var seqOffset1 = result.offset

          val data0   = dv.data
          val stride0 = dv.stride
          var offset0 = dv.offset
          val end0    = dv.endOffset

          // Perform demultiplexing.
          // TODO: Could make this  slightly faster.
          while (offset0 != end0) {
            var offset1 = seqOffset1
            var i = 0
            while (i < sequenceLength) {
              data1(offset1) = data0(offset0)
              offset0 += stride0
              offset1 += seqStride1
              i       += 1
            }
            seqOffset1 += valStride1
          }
        }
    */
    /*

        /**
         * We explicitly constraint the types T and U as hints. This allows the
         * JVM to transform virtual into explicit function calls.
         *
         * @param fn Function to execute on each item.
         * @param fn2 Will automatically be resolved from namespace fn.
         * @tparam V Parent UFunc object type.
         * @tparam W The subfunction to execute.
         */
        @inline
        def fastMapFn[V <: UFunc, W <: UFunc.UImpl[V, T, T]](fn: V)
                                                            (implicit fn2: W, classTag: ClassTag[T])
        : DenseVector[T] = dv.fastMap(fn2.apply)

        /**
         * We explicitly constraint the types T and U as hints. This allows the
         * JVM to transform virtual into explicit function calls.
         *
         * @param fn Function to execute on each item.
         * @param fn2 Will automatically be resolved from namespace fn.
         * @tparam V Parent UFunc object type.
         * @tparam W The subfunction to execute.
         */
        @inline
        def fastMapFn[V <: UFunc, W <: UFunc.UImpl2[V, T, X, T], X](fn: V, v2: X)
                                                                   (implicit fn2: W, classTag: ClassTag[T])
        : DenseVector[T] = dv.fastMap(fn2(_, v2))

        /**
         * We explicitly constraint the types T and U as hints. This allows the
         * JVM to transform virtual into explicit function calls.
         *
         * @param fn Function to execute on each item.
         * @param fn2 Will automatically be resolved from namespace fn.
         * @tparam V Parent UFunc object type.
         * @tparam W The subfunction to execute.
         */
        @inline
        def fastMapFn[V <: UFunc, W <: UFunc.UImpl3[V, T, X, Y, T], X, Y](fn: V, v2: X, v3: Y)
                                                                         (implicit fn2: W, classTag: ClassTag[T])
        : DenseVector[T] = dv.fastMap(fn2(_, v2, v3))

        /**
         * We explicitly constraint the types T and U as hints. This allows the
         * JVM to transform virtual into explicit function calls.
         *
         * @param fn Function to execute on each item.
         * @param fn2 Will automatically be resolved from namespace fn.
         * @tparam V Parent UFunc object type.
         * @tparam W The subfunction to execute.
         */
        @inline
        def fastMapFn[V <: UFunc, W <: UFunc.UImpl4[V, T, X, Y, Z, T], X, Y, Z](fn: V, v2: X, v3: Y, v4: Z)
                                                                               (implicit fn2: W, classTag: ClassTag[T])
        : DenseVector[T] = dv.fastMap(fn2(_, v2, v3, v4))
    */

    /*
    @inline
    def fastToSparse(implicit classTag: ClassTag[T], zero: Zero[T])
    : SparseVector[T] = {
      val data0    = dv.data
      val stride0  = dv.stride
      val used1    = dv.length
      val indices1 = Array.ofDim[Int](used1)
      val data1    = Array.ofDim[T](used1)

      // TODO: Could avoid some java range checks.
      var offset0 = dv.offset
      var offset1 = 0
      while (offset1 < used1) {
        data1(offset1) = data0(offset0)
        offset0 += stride0
        indices1(offset1) = offset1
        offset1  += 1
      }
      new SparseVector(indices1, data1, used1)
    }
    */

    /*

      @inline
      def multiplex(sequenceLength: Int)
                   (implicit classTag: ClassTag[T], zero: Zero[T])
      : DenseVector[T] = {
        val result = DenseVector.zeros[T](dv.length)
        multiplex(sequenceLength, result)
        result
      }

      // TODO: Add optimized versions to DMat and Array!
      @inline
      def multiplex(sequenceLength: Int, result: DenseVector[T]): Unit = {
        require(dv.length == result.length)

        val noSequences = dv.length / sequenceLength
        assume(noSequences * sequenceLength == dv.length)

        // Fetch frequently used values.
        val data1   = result.data
        val stride1 = result.stride
        var offset1 = result.offset
        val end1    = result.endOffset

        val data0      = dv.data
        val valStride0 = dv.stride
        val seqStride0 = dv.stride * noSequences
        var seqOffset0 = dv.offset

        // Perform multiplexing.
        // TODO: Could make this  slightly faster.
        while (offset1 != end1) {
          var offset0 = seqOffset0
          var i = 0
          while (i < sequenceLength) {
            data1(offset1) = data0(offset0)
            offset1 += stride1
            offset0 += seqStride0
            i       += 1
          }
          seqOffset0 += valStride0
        }
      }

      @inline
      def intersects(other: DenseVector[T]): Boolean = {
        if ((dv.data ne other.data) || dv.length == 0 || other.length == 0) {
          false
        }
        else if (dv.offset >= other.endOffset || dv.endOffset <= other.offset) {
          false
        }
        else if (dv.stride == 1 || other.stride == 1) {
          true
        }
        else {
          val a = dv.offsets
          val b = other.offsets
          a.intersect(b).nonEmpty
        }
      }
      */

    /*
    final def transformActiveEx[U](other: SparseVector[U])
                                  (fn0: (T, U) => T)
    : Unit = {
      debug_req(dv.length == other.length)

      val data1    = other.data
      val indices1 = other.index
      val end1     = other.used
      var offset1  = 0

      while (offset1 < end1) {
        val i = indices1(offset1)
        // TODO: Could do this faster.
        dv.unsafeUpdate(i, fn0(dv.unsafeValueAt(i), data1(offset1)))
        offset1 += 1
      }
    }
    */

    /*
    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The subfunction to execute.
     */
    @inline
    def transformFn[V <: UFunc, W <: UFunc.UImpl[V, T, T]](fn: V)
                                                          (implicit fn2: W)
    : Unit = transform(fn2.apply)

    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The subfunction to execute.
     */
    @inline
    def transformFn[V <: UFunc, W <: UFunc.UImpl2[V, T, X, T], X](fn: V, v2: X)
                                                                 (implicit fn2: W)
    : Unit = transform(fn2(_, v2))

    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The subfunction to execute.
     */
    @inline
    def transformFn[V <: UFunc, W <: UFunc.UImpl3[V, T, X, Y, T], X, Y](fn: V, v2: X, v3: Y)
                                                                       (implicit fn2: W)
    : Unit = transform(fn2(_, v2, v3))

    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The subfunction to execute.
     */
    @inline
    def transformFn[V <: UFunc, W <: UFunc.UImpl4[V, T, X, Y, Z, T], X, Y, Z](fn: V, v2: X, v3: Y, v4: Z)
                                                                             (implicit fn2: W)
    : Unit = transform(fn2(_, v2, v3, v4))
    */

  }
  */

  //type WBuffer = DenseVector[DVec]

  //type WBufferSize = DenseVector[Int]

  /*
  implicit class WBufferSizeFunctions(wbs: WBufferSize) {

    def +?(other: WBufferSize): WBufferSize = {
      wbs.zip(other)(_ +? _)
    }

  }*/

  //type LooseBatch = Seq[Sample]

  //type LooseBatchAct = Seq[SampleActLike]

  //type DLooseBatchAct = Seq[DSampleAct]

  //type SLooseBatchAct = Seq[SSampleAct]

  //type DoPredictResultS = (SampleTensor, Any)

  //type DoPredictResultB = (BatchTensor, Any)

  //type DoComputeCostResultS = (Real, SampleTensor, Any)

  //type DoComputeCostResultB = (Real, Tensor, Any)

  //type PredictCallbackS = (Module, ComputeMode, SampleTensor, SampleTensor, Any) => Unit

  type OnEnterPredict = (Module, Tensor, Tensor) => Boolean

  type OnLeavePredict = (Module, Tensor, Tensor, Tensor, PredictContext) => Unit

  //type PredictInvCallbackS = (Module, ComputeMode, SampleTensor) => Unit

  //type ComputeCostCallbackS = (Module, ComputeMode, SampleTensor, SampleTensor, Any, SampleTensor) => Unit

  type OnEnterEvaluate = (Module, Tensor, Tensor, Real) => Real

  type OnLeaveEvaluate = (Module, Tensor, Tensor, Tensor, PredictContext, Real) => Real

  type OnEnterDeriveGradients = (Module, Tensor, Tensor, Tensor, PredictContext, NextError) => Unit

  type OnLeaveDeriveGradients = OnEnterDeriveGradients

  final val idleOnEnterPredict
  : OnEnterPredict = (module:    Module,
                      input:     Tensor,
                      reference: Tensor) => true

  final val idleOnLeavePredict
  : OnLeavePredict = (module:    Module,
                      input:     Tensor,
                      reference: Tensor,
                      output:    Tensor,
                      context:   PredictContext) => {}

  final val idleOnEnterDeriveInputError
  : OnEnterDeriveGradients = (module:    Module,
                              input:     Tensor,
                              reference: Tensor,
                              output:    Tensor,
                              context:   PredictContext,
                              error:     NextError) => {}

  final val idleOnLeaveDeriveInputError
  : OnLeaveDeriveGradients = idleOnEnterDeriveInputError


  final implicit class ValidatorFunctions(v: Validator) {

    def apply(model: Module, batchPool: BatchPool)
    : ValidationScore = {
      var result  = ValidationScore.zero
      var context = batchPool.draw()
      while (true) {
        val batch = context.batch
        if (batch == null) {
          return result
        }
        val p = model.predict(Inference(), batch).dropIntermediates()
        result += apply(p)
        context.close()
        context = batchPool.draw()
      }
      result
    }

    def apply(prediction: Prediction)
    : ValidationScore = v.apply(prediction.reference, prediction.output)

    def apply(predictions: TraversableOnce[Prediction])
    : ValidationScore = predictions.foldLeft(
      ValidationScore.zero
    )(_ + apply(_))

    def apply[T <: Prediction](predictions: Array[T])
    : ValidationScore = ArrayEx.foldLeft(
      ValidationScore.zero, predictions
    )(_ + apply(_))

  }

}
