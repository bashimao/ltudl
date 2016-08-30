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
import edu.latrobe.io.graph._
import edu.latrobe.time._
import scala.collection._
import scala.util.hashing._
import spire.implicits._

/**
 * Named objects that support binding and updating.
 *
 * Common base for layers and models.
 *
 * doFunctions are special. They only operate in the context of the
 * current object. They should not be called directly unless you
 * know what you do.
 */
abstract class Module
  extends InstanceEx[ModuleBuilder]
    with ParameterizedInstance {

  /**
   * Must be overwritten with a constructor argument.
   */
  def inputHints
  : BuildHints

  final val inputLayoutHint
  : TensorLayout = inputHints.layout

  final val inputSizeHint
  : Size = inputLayoutHint.size

  /**
   * Can be implemented arbitrarily. But lazy val is probably your friend here
   * because this is touched during each build and it should somewhat not
   * depend on too much other stuff. (see implementation for Layer, to see why)
   */
  def outputHints
  : BuildHints

  /**
   * Must be overwritten with a constructor argument.
   */
  def weightBufferBuilder
  : ValueTensorBufferBuilder

  final val handle
  : String = builder.handle


  // ---------------------------------------------------------------------------
  //   Statistics
  // ---------------------------------------------------------------------------
  /**
   * Should be implemented as a lazy val.
   *
   * @return Number of neurons in module.
   */
  def noNeurons
  : Long


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  /**
    * Should be implemented as a lazy val.
    */
  @transient
  final lazy val weightBuffer
  : ValueTensorBuffer = weightBufferBuilder.result()

  /**
    * Typically you want to override this with a lazy val.
    */
  def weightReferences
  : Set[LabeledBufferReference]

  /**
    * Reinitializes the model.
    */
  def reset(initializer: Initializer)
  : Unit

  /**
    * Synchronizes any native hardware buffers and JVM buffers.
    */
  def refresh()
  : Unit

  final def extractWeights()
  : Array[Array[Real]] = {
    val n = noNeurons
    require(n < Int.MaxValue)
    ArrayEx.tabulate(n.toInt)(extractWeightsFor(_))
  }

  def extractWeightsFor(neuronNo
    : Long): Array[Real]

  final def extractWeightsFor(neuronNos: Seq[Long])
  : SortedMap[Long, Array[Real]] = {
    val builder = SortedMap.newBuilder[Long, Array[Real]]
    neuronNos.foreach(
      neuronNo => builder += Tuple2(neuronNo, extractWeightsFor(neuronNo))
    )
    builder.result()
  }


  // ---------------------------------------------------------------------------
  //    Traversal related.
  // ---------------------------------------------------------------------------
  /**
   * Traverses forward through all modules.
   *
   * @param callbackFn Executed for each encountered module.
   */
  final def touch(callbackFn: Module => Unit)
  : Unit = {
    callbackFn(this)
    doTouch(callbackFn)
  }

  protected def doTouch(callbackFn: Module => Unit)
  : Unit


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  /**
    * Modules that require access to their original input throughout the entire
    * forward propagation cycle should set this to true. This is only relevant
    * for containers.
    */
  def requiresMaintainingInputDuringForwardPropagation
  : Boolean

  /**
   * Predicts output activations based on the current state of the element.
   *
   * @param input [in] Input activations.
   * @return Estimated output activations.
   */
  final def predict(mode:      Mode,
                    input:     Tensor,
                    reference: Tensor)
  : BackpropagationContext = predict(
    mode,
    input,
    reference,
    idleOnEnterPredict,
    idleOnLeavePredict
  )

  /**
   * Predicts output activations based on the current state of the element and
   * executes a callback.
   *
   * @param input [in] Input activations.
   * @param onLeave Called for each prediction of a sub-module.
   * @return Processed output activations.
   */
  final def predict(mode:      Mode,
                    input:     Tensor,
                    reference: Tensor,
                    onEnter:   OnEnterPredict,
                    onLeave:   OnLeavePredict)
  : BackpropagationContext = {
    var value     = Real.zero
    val bpBuilder = BackpropagationContext.newBuilder(input, reference)

    def onEnterEx(module:    Module,
                  input:     Tensor,
                  reference: Tensor)
    : Boolean = {
      // Block the tensor if it is required throughout the BPROP or FPROP.
      if (module.requiresMaintainingInputDuringForwardPropagation) {
        bpBuilder.block(input)
      }
      else {
        bpBuilder.block(null)
      }

      // Allow in-place modification if not blocked and not used somewhere else.
      var inPlaceAllowed = onEnter(module, input, reference)
      if (inPlaceAllowed) {
        inPlaceAllowed = !bpBuilder.isBlocked(input)
      }
      if (inPlaceAllowed) {
        inPlaceAllowed = !bpBuilder.requiresMaintaining(input)
      }
      inPlaceAllowed
    }

    def onLeaveEx(module:    Module,
                  input:     Tensor,
                  reference: Tensor,
                  output:    Tensor,
                  context:   PredictContext)
    : Unit = {
      // Update cost, call leave callback and unblock top-most input tensor.
      value += module.doEvaluate(input, reference, output, context)
      onLeave(module, input, reference, output, context)
      bpBuilder.unblock()

      // Add tensors we need for backprop to the list.
      // INP
      var needInp = false
      // If backpropagation is supported in the selected mode.
      if (mode.supportsBackpropagation) {
        module.backpropagationRequirementsForInput match {
          case TensorDependency.Required =>
            // If required for backpropagation by the current module and the mode of operation
            needInp = true
          case _ =>
        }
      }
      if (!needInp) {
        // However, if another module requires this tensor later or if we depend somehow on a tensor that must be kept.
        needInp = bpBuilder.requiresMaintaining(input)
      }
      bpBuilder.stash(if (needInp) input else null)
      if (needInp && logger.isTraceEnabled) {
        val outSize = StringEx.render(input.layout.noValues * Real.size, 1024)
        val totSize = bpBuilder.stashSize.mapValues(
          noValues => StringEx.render(noValues * Real.size, 1024)
        )
        logger.trace(s"Stashed input!  $module, ${input.platform} + $outSize | $totSize")
      }

      // REF
      bpBuilder.stash(reference)

      // OUT
      // If backpropagation is supported in the selected mode.
      var needOut = false
      if (mode.supportsBackpropagation) {
        needOut = module.backpropagationRequirementsForOutput match {
          case TensorDependency.Required =>
            true
          case TensorDependency.NotRequired =>
            false
          case _ =>
            !needInp
        }
      }
      bpBuilder.stash(if (needOut) output else null)
      if (needOut && logger.isTraceEnabled) {
        val outSize = StringEx.render(input.layout.noValues * Real.size, 1024)
        val totSize = bpBuilder.stashSize.mapValues(
          noValues => StringEx.render(noValues * Real.size, 1024)
        )
        logger.trace(s"Stashed output! $module, ${output.platform} + $outSize | $totSize")
      }

      // Deallocate input tensors that we no longer need.
      if (!input.dependsOn(output)) {
        if (!bpBuilder.isBlocked(input)) {
          if (!bpBuilder.requiresMaintaining(input)) {
            input.close()
          }
        }
      }

      // CTX
      bpBuilder.stash(context)
    }

    val output = predictEx(
      mode,
      input, reference,
      onEnterEx, onLeaveEx
    )

    bpBuilder.result(mode, output, value)
  }

  final def predict(mode:  Mode,
                    batch: Batch)
  : BackpropagationContext = {
    predict(
      mode,
      batch,
      idleOnEnterPredict,
      idleOnLeavePredict
    )
  }

  final def predict(mode:    Mode,
                    batches: Traversable[Batch])
  : Traversable[BackpropagationContext] = {
    batches.map(
      predict(mode, _)
    )
  }

  final def predict(mode:    Mode,
                    batches: Iterable[Batch])
  : Iterable[BackpropagationContext] = {
    batches.map(
      predict(mode, _)
    )
  }

  final def predict(mode:    Mode,
                    batches: Seq[Batch])
  : Seq[BackpropagationContext] = {
    batches.map(
      predict(mode, _)
    )
  }

  final def predict(mode:    Mode,
                    batches: IndexedSeq[Batch])
  : IndexedSeq[BackpropagationContext] = {
    batches.map(
      predict(mode, _)
    )
  }

  final def predict(mode:    Mode,
                    batches: Array[Batch])
  : Array[BackpropagationContext] = {
    ArrayEx.map(
      batches
    )(predict(mode, _))
  }

  final def predict(mode:    Mode,
                    batch:   Batch,
                    onEnter: OnEnterPredict,
                    onLeave: OnLeavePredict)
  : BackpropagationContext = {
    predict(
      mode,
      batch.input,
      batch.output,
      onEnter,
      onLeave
    )
  }

  final def predict(mode:    Mode,
                    batches: Traversable[Batch],
                    onEnter: OnEnterPredict,
                    onLeave: OnLeavePredict)
  : Traversable[BackpropagationContext] = {
    batches.map(
      batch => predict(
        mode,
        batch.input,
        batch.output,
        onEnter,
        onLeave
      )
    )
  }

  final def predict(mode:    Mode,
                    batches: Iterable[Batch],
                    onEnter: OnEnterPredict,
                    onLeave: OnLeavePredict)
  : Iterable[BackpropagationContext] = {
    batches.map(batch => predict(
      mode,
      batch.input,
      batch.output,
      onEnter,
      onLeave
    ))
  }

  final def predict(mode:    Mode,
                    batches: Seq[Batch],
                    onEnter: OnEnterPredict,
                    onLeave: OnLeavePredict)
  : Seq[BackpropagationContext] = {
    batches.map(batch => predict(
      mode,
      batch.input,
      batch.output,
      onEnter,
      onLeave
    ))
  }

  final def predict(mode:    Mode,
                    batches: IndexedSeq[Batch],
                    onEnter: OnEnterPredict,
                    onLeave: OnLeavePredict)
  : IndexedSeq[BackpropagationContext] = {
    batches.map(batch => predict(
      mode,
      batch.input,
      batch.output,
      onEnter,
      onLeave
    ))
  }

  final def predict(mode:    Mode,
                    batches: Array[Batch],
                    onEnter: OnEnterPredict,
                    onLeave: OnLeavePredict)
  : Array[BackpropagationContext] = {
    ArrayEx.map(
      batches
    )(batch => predict(mode, batch.input, batch.output, onEnter, onLeave))
  }

  // TODO: Find better solution for this.
  final protected[blaze] def predictEx(mode:      Mode,
                                       input:     Tensor,
                                       reference: Tensor,
                                       onEnter:   OnEnterPredict,
                                       onLeave:   OnLeavePredict)
  : Tensor = {
    val clock = if (logger.isTraceEnabled) Stopwatch() else null

    val inPlaceAllowed = onEnter(this, input, reference)
    val (output, context) = doPredict(
      mode,
      inPlaceAllowed, input, reference,
      onEnter, onLeave
    )
    onLeave(this, input, reference, output, context)

    if (clock != null) {
      val ipa = if (inPlaceAllowed) "ipa" else "   "
      logger.trace(
        f"$clock%s => predict($ipa, ${input.platform}%-4s) => $this%s"
      )
    }
    output
  }

  /**
   * Predicts output activations based on the current state of the element and
   * executes a callback.
 *
   * @param input [in] Input activations.
   * @param onEnter Called for each prediction of a sub-module.
   * @return Processed output activations.
   */
  protected def doPredict(mode:           Mode,
                          inPlaceAllowed: Boolean,
                          input:          Tensor,
                          reference:      Tensor,
                          onEnter:        OnEnterPredict,
                          onLeave:        OnLeavePredict)
  : (Tensor, PredictContext)

  final def project(mode: Mode, batch: Batch)
  : Batch = {
    val prediction = predict(mode, batch).dropIntermediates()
    batch.derive(prediction.output)
  }

  final def predictInv(context: BackpropagationContext)
  : Tensor = predictInv(context, idleOnLeavePredict)

  final def predictInv(contexts: Traversable[BackpropagationContext])
  : Traversable[Tensor] = contexts.map(predictInv)

  final def predictInv(contexts: Iterable[BackpropagationContext])
  : Iterable[Tensor] = contexts.map(predictInv)

  final def predictInv(contexts: Seq[BackpropagationContext])
  : Seq[Tensor] = contexts.map(predictInv)

  final def predictInv(contexts: IndexedSeq[BackpropagationContext])
  : IndexedSeq[Tensor] = contexts.map(predictInv)

  final def predictInv(contexts: Array[BackpropagationContext])
  : Array[Tensor] = ArrayEx.map(contexts)(predictInv)

  /**
    * Predicts input activations based on the current state of the layer.
    * (not possible with all layer types)
    *
    * @param context [in] Output activations.
    * @return Estimated input activations.
    */
  final def predictInv(context: BackpropagationContext,
                       onLeave: OnLeavePredict)
  : Tensor = {
    def onLeaveEx(module:    Module,
                  input:     Tensor,
                  reference: Tensor,
                  _output:   Tensor,
                  context:   PredictContext)
    : Unit = {
      onLeave(module, input, reference, _output, context)
      // TODO: Does not really work correctly if using tables.
      if ((_output ne input) && (_output ne _output)) {
        input.close()
      }
    }
    val _contexts = mutable.Stack.concat(context.contexts)
    predictInvEx(context.output, onLeaveEx, _contexts)
  }

  final def predictInv(contexts: Traversable[BackpropagationContext],
                       onLeave:  OnLeavePredict)
  : Traversable[Tensor] = contexts.map(predictInv(_, onLeave))

  final def predictInv(contexts: Iterable[BackpropagationContext],
                       onLeave:  OnLeavePredict)
  : Iterable[Tensor] = contexts.map(predictInv(_, onLeave))

  final def predictInv(contexts: Seq[BackpropagationContext],
                       onLeave:  OnLeavePredict)
  : Seq[Tensor] = contexts.map(predictInv(_, onLeave))

  final def predictInv(contexts: IndexedSeq[BackpropagationContext],
                       onLeave:  OnLeavePredict)
  : IndexedSeq[Tensor] = contexts.map(predictInv(_, onLeave))

  final def predictInv(contexts: Array[BackpropagationContext],
                       onLeave:  OnLeavePredict)
  : Array[Tensor] = ArrayEx.map(contexts)(predictInv(_, onLeave))

  final protected[blaze] def predictInvEx(output:   Tensor,
                                          onLeave:  OnLeavePredict,
                                          contexts: mutable.Stack[PredictContext])
  : Tensor = {
    val clock = if (logger.isTraceEnabled) Stopwatch() else null

    val context = contexts.pop()
    val input   = doPredictInv(output, context, onLeave, contexts)
    onLeave(this, input, null, output, context)

    if (clock != null) {
      logger.trace(f"$clock%s => predictInv(${output.platform}%-4s) => $this%s")
    }
    input
  }

  /**
   * Performs inverse prediction (output to input). Not possible for all modules.
   *
   * @param output [in] Output activations.
   * @param onLeave [in] Function to be called after each prediction.
   * @return Input activations.
   */
  protected def doPredictInv(output:   Tensor,
                             context:  PredictContext,
                             onLeave:  OnLeavePredict,
                             contexts: mutable.Stack[PredictContext])
  : Tensor
  /*
  // Start with what we have.
  var contexts = prediction.contexts

  // Traverse backward through layers.
  var out = prediction.output
  traverseBackward(
    mode,
    module => {
      // Get context.
      val ctx = contexts.head; contexts = contexts.tail

      // Do a local prediction.
      val inp = module.doPredictInv(mode, out, ctx)
      if (callback != null) {
        callback(this, mode, inp, out, ctx)
      }

      // Deallocate output if:
      // 1. It is not identical to the input.
      // 2. It was not the original output.
      if (!(out eq inp) && !(out eq prediction.output)) {
        out.deallocate()
      }

      // Next!
      out = inp
    },
    module => {},
    () => throw new UnsupportedOperationException,
    () => throw new UnsupportedOperationException,
    () => throw new UnsupportedOperationException,
    () => throw new UnsupportedOperationException
  )
  out
  */
  /*
  /**
   * Performs inverse prediction (output to input). Not possible for all modules.
   *
   * @param mode [in] Mode of operation.
   * @param output [in] Output activations.
   * @param callback [in] Function to be called after each prediction.
   * @return Input activations.
   */
  protected def doPredictInv(mode:     ComputeMode,
                             output:   SampleTensor,
                             context:  Any,
                             callback: PredictInvCallbackS)
  : SampleTensor*/


  // ---------------------------------------------------------------------------
  //    Cost/Gradient computation related.
  // ---------------------------------------------------------------------------

  /**
   * Modules that require access to the original layer input at any stage during
   * backpropagation should set this to true.
   */
  //def requiresInputForBackpropagation(mode: ComputeMode): Boolean

  /**
   * Modules that require access to the untainted layer output at any stage
   * during backpropagation should set this to true.
   */
  //def requiresOutputForBackpropagation(mode: ComputeMode): Boolean

  /*
  final def evaluate(mode:      OperationMode,
                     input:     Tensor,
                     reference: Tensor)
  : EvaluationResult = evaluate(
    mode, input, reference, idleOnEnterEvaluate, idleOnLeaveEvaluate
  )

  final def evaluate(mode:      OperationMode,
                     input:     Tensor,
                     reference: Tensor,
                     onEnter:   OnEnterEvaluate,
                     onLeave:   OnLeaveEvaluate)
  : EvaluationResult = {

    val prediction = predict(mode, input, reference, onEnterEx, onLeaveEx)
    EvaluationResult(prediction, value)
  }

  final def evaluate(mode: OperationMode, batch: Batch)
  : EvaluationResult = evaluate(
    mode, batch, idleOnEnterEvaluate, idleOnLeaveEvaluate
  )

  final def evaluate(mode: OperationMode, batches: Traversable[Batch])
  : Traversable[EvaluationResult] = batches.map(evaluate(mode, _))

  final def evaluate(mode: OperationMode, batches: Iterable[Batch])
  : Iterable[EvaluationResult] = batches.map(evaluate(mode, _))

  final def evaluate(mode: OperationMode, batches: Seq[Batch])
  : Seq[EvaluationResult] = batches.map(evaluate(mode, _))

  final def evaluate(mode: OperationMode, batches: IndexedSeq[Batch])
  : IndexedSeq[EvaluationResult] = batches.map(evaluate(mode, _))

  final def evaluate(mode: OperationMode, batches: Array[Batch])
  : Array[EvaluationResult] = ArrayEx.map(batches)(evaluate(mode, _))

  final def evaluate(mode: OperationMode, batches: DenseVector[Batch])
  : Array[EvaluationResult] = VectorEx.map(batches)(evaluate(mode, _))

  final def evaluate(mode: OperationMode, batches: DenseMatrix[Batch])
  : Array[EvaluationResult] = MatrixEx.map(batches)(evaluate(mode, _))

  final def evaluate(mode:    OperationMode,
                     batch:   Batch,
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : EvaluationResult = evaluate(
    mode, batch.input, batch.output, onEnter, onLeave
  )

  final def evaluate(mode:    OperationMode,
                     batches: Traversable[Batch],
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : Traversable[EvaluationResult] = batches.map(
    evaluate(mode, _, onEnter, onLeave)
  )

  final def evaluate(mode:    OperationMode,
                     batches: Iterable[Batch],
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : Iterable[EvaluationResult] = batches.map(
    evaluate(mode, _, onEnter, onLeave)
  )


  final def evaluate(mode:    OperationMode,
                     batches: Seq[Batch],
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : Seq[EvaluationResult] = batches.map(evaluate(mode, _, onEnter, onLeave))

  final def evaluate(mode:    OperationMode,
                     batches: IndexedSeq[Batch],
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : IndexedSeq[EvaluationResult] = batches.map(
    evaluate(mode, _, onEnter, onLeave)
  )

  final def evaluate(mode:    OperationMode,
                     batches: Array[Batch],
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : Array[EvaluationResult] = ArrayEx.map(batches)(
    evaluate(mode, _, onEnter, onLeave)
  )

  final def evaluate(mode:    OperationMode,
                     batches: DenseVector[Batch],
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : Array[EvaluationResult] = VectorEx.map(batches)(
    evaluate(mode, _, onEnter, onLeave)
  )

  final def evaluate(mode:    OperationMode,
                     batches: DenseMatrix[Batch],
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : Array[EvaluationResult] = MatrixEx.map(batches)(
    evaluate(mode, _, onEnter, onLeave)
  )
  */

  /*
  final def evaluate(mode: OperationMode, batches: BatchPool)
  : Cost = evaluate(mode, batches, idleOnEnterEvaluate, idleOnLeaveEvaluate)

  final def evaluate(mode:    OperationMode,
                     batches: BatchPool,
                     onEnter: OnEnterEvaluate,
                     onLeave: OnLeaveEvaluate)
  : Cost = batches.foldLeft(Cost.zero)((cost, batch) => {
    cost + evaluate(mode, batch, onEnter, onLeave).dropPrediction()
  })
  */

  /**
   * This callback is used for estimating the cost of for the current module.
   * The callback is automatically called by computeCost after each prediction.
   * So you only have to call computations in which are not triggered by that.
   *
   * @param input [in] Input activations of the module.
   * @param reference [in] Activations to compare against (only for cost functions!)
   * @return The cost contribution and prediction of this module. (avg. cost per sample in batch)
   */
  // THIS HAS JUST A LOCAL SCOPE!
  protected def doEvaluate(input:     Tensor,
                           reference: Tensor,
                           output:    Tensor,
                           context:   PredictContext)
  : Real


  // ---------------------------------------------------------------------------
  //    Derive input error (without given error)
  // ---------------------------------------------------------------------------
  /**
    * Modules that require access to the original layer input at any stage
    * during backpropagation should set this to true.
    *
    * We use this flag to determine whether we should keep or discard a certain
    * input. Thus allowing us to save memory.
    *
    */
  def backpropagationRequirementsForInput
  : TensorDependency

  /**
    * Modules that require access to the untainted layer output at any stage
    * during backpropagation must set this to true.
    *
    * We use this flag to determine whether we should keep or discard a certain
    * input. Thus allowing us to save memory.
    */
  def backpropagationRequirementsForOutput
  : TensorDependency

  /*
  // TODO: Handle this better to make memory consumption tunable.
  final lazy val prefersInputForBackpropagation: Boolean = {
    val inpDep = dependenceOnInputForBackpropagation
    val outDep = dependenceOnOutputForBackpropagation
    inpDep.level < outDep.level
  }

  // TODO: Handle this better to make memory consumption tunable.
  final lazy val prefersOutputForBackpropagation
  : Boolean = !prefersInputForBackpropagation
  */

  /*
  final def deriveInputError(prediction: PredictionEx)
  : NextError = deriveInputError(
    prediction,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveInputError(predictions: Traversable[PredictionEx])
  : Traversable[NextError] = predictions.map(deriveInputError)

  final def deriveInputError(predictions: Iterable[PredictionEx])
  : Iterable[NextError] = predictions.map(deriveInputError)

  final def deriveInputError(predictions: Seq[PredictionEx])
  : Seq[NextError] = predictions.map(deriveInputError)

  final def deriveInputError(predictions: IndexedSeq[PredictionEx])
  : IndexedSeq[NextError] = predictions.map(deriveInputError)

  final def deriveInputError(predictions: Array[PredictionEx])
  : Array[NextError] = predictions.fastMap(deriveInputError)

  final def deriveInputError(predictions: DenseVector[PredictionEx])
  : DenseVector[NextError] = predictions.fastMap(deriveInputError)

  final def deriveInputError(predictions: DenseMatrix[PredictionEx])
  : DenseMatrix[NextError] = predictions.fastMap(deriveInputError)

  final def deriveInputError(prediction: PredictionEx,
                             onEnter:    OnEnterDeriveInputError,
                             onLeave:    OnLeaveDeriveInputError)
  : NextError = deriveInputErrorEx(prediction, onEnter, onLeave)

  final def deriveInputError(predictions: Traversable[PredictionEx],
                             onEnter:     OnEnterDeriveInputError,
                             onLeave:     OnLeaveDeriveInputError)
  : Traversable[NextError] = predictions.map(
    deriveInputError(_, onEnter, onLeave)
  )

  final def deriveInputError(predictions: Iterable[PredictionEx],
                             onEnter:     OnEnterDeriveInputError,
                             onLeave:     OnLeaveDeriveInputError)
  : Iterable[NextError] = predictions.map(
    deriveInputError(_, onEnter, onLeave)
  )

  final def deriveInputError(predictions: Seq[PredictionEx],
                             onEnter:     OnEnterDeriveInputError,
                             onLeave:     OnLeaveDeriveInputError)
  : Seq[NextError] = predictions.map(
    deriveInputError(_, onEnter, onLeave)
  )

  final def deriveInputError(predictions: IndexedSeq[PredictionEx],
                             onEnter:     OnEnterDeriveInputError,
                             onLeave:     OnLeaveDeriveInputError)
  : IndexedSeq[NextError] = predictions.map(
    deriveInputError(_, onEnter, onLeave)
  )

  final def deriveInputError(predictions: Array[PredictionEx],
                             onEnter:     OnEnterDeriveInputError,
                             onLeave:     OnLeaveDeriveInputError)
  : Array[NextError] = predictions.fastMap(
    deriveInputError(_, onEnter, onLeave)
  )

  final def deriveInputError(predictions: DenseVector[PredictionEx],
                             onEnter:     OnEnterDeriveInputError,
                             onLeave:     OnLeaveDeriveInputError)
  : DenseVector[NextError] = predictions.fastMap(
    deriveInputError(_, onEnter, onLeave)
  )

  final def deriveInputError(predictions: DenseMatrix[PredictionEx],
                             onEnter:     OnEnterDeriveInputError,
                             onLeave:     OnLeaveDeriveInputError)
  : DenseMatrix[NextError] = predictions.fastMap(
    deriveInputError(_, onEnter, onLeave)
  )
  */


  // ---------------------------------------------------------------------------
  //    Derive input error (with given error)
  // ---------------------------------------------------------------------------
  /*
  final def deriveInputError(prediction: PredictionEx,
                             error:      Tensor)
  : NextError = deriveInputError(
    prediction,
    error,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveInputError(prediction: PredictionEx,
                             error:      Tensor,
                             onEnter:    OnEnterDeriveInputError,
                             onLeave:    OnLeaveDeriveInputError)
  : NextError = deriveInputErrorEx(
    prediction,
    NextError(() => error.copy),
    onEnter,
    onLeave
  )*/

  /*
  final private def deriveInputErrorEx(prediction: PredictionEx,
                                       onEnter:    OnEnterDeriveInputError,
                                       onLeave:    OnLeaveDeriveInputError)
  : NextError = deriveInputErrorEx(
    prediction,
    NextError(prediction.output.allocateSiblingAndClear),
    onEnter,
    onLeave
  )

  final private def deriveInputErrorEx(prediction: PredictionEx,
                                       error:      NextError,
                                       onEnter:    OnEnterDeriveInputError,
                                       onLeave:    OnLeaveDeriveInputError)
  : NextError = {

  }
  */


  // ---------------------------------------------------------------------------
  //    Derive gradient (without given input error, without sink)
  // ---------------------------------------------------------------------------

  /*
  final def deriveGradients(prediction: PredictionEx)
  : ParameterBuffer = deriveGradients(
    prediction, idleOnEnterDeriveInputError, idleOnLeaveDeriveInputError
  )

  final def deriveGradients(prediction: PredictionEx,
                            onEnter:    OnEnterDeriveInputError,
                            onLeave:    OnLeaveDeriveInputError)
  : (SortedMap[Int, Tensor], ParameterBuffer) = {
    val gradients = ParameterBuffer.zeros(weightsLayout)
    gradients.groups.foreach(sink => {
      deriveGradients(prediction, sink, onEnter, onLeave).disposeThorough()
    })
    gradients
  }
   */

  // ---------------------------------------------------------------------------
  //    Derive gradient (with given input error, without sink)
  // ---------------------------------------------------------------------------
  /*
  final def deriveGradients(prediction: PredictionEx, error: Tensor)
  : (NextError, ParameterBuffer) = deriveGradients(
    prediction,
    error,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveGradients(prediction: PredictionEx,
                            error:      Tensor,
                            onEnter:    OnEnterDeriveInputError,
                            onLeave:    OnLeaveDeriveInputError)
  : (NextError, ParameterBuffer) = {
    val gradients = ParameterBuffer.zeros(weightsLayout)
    val errors    = gradients.groups.map(
      kv => kv._1 -> deriveGradients(prediction, error, kv, onEnter, onLeave)
    )
    errors -> gradients
  }

  final def deriveGradients(predictions: Iterable[PredictionEx],
                            errors:      Iterable[Tensor],
                            onEnter:     OnEnterDeriveInputError,
                            onLeave:     OnLeaveDeriveInputError)
  : ParameterBuffer = {
    val layout    = weightsLayout
    val gradients = ParameterBuffer.zeros(layout)
    var noSamples = 0

    predictions.fastForeachEx(errors)((p, e) => {
      val tmp = deriveGradients(p, e, onEnter, onLeave)
      val n = p.input.noSamples
      noSamples += n
      lerp.inPlace(gradients, tmp, n / Real(noSamples))
    })
    gradients
  }

  final def deriveGradients(predictions: Array[PredictionEx],
                            errors:      Array[Tensor],
                            onEnter:     OnEnterDeriveInputError,
                            onLeave:     OnLeaveDeriveInputError)
  : ParameterBuffer = {
    val layout    = weightsLayout
    val gradients = ParameterBuffer.zeros(layout)
    var noSamples = 0

    predictions.fastForeachEx(errors)((p, e) => {
      val tmp = deriveGradients(p, e, onEnter, onLeave)
      val n = p.input.noSamples
      noSamples += n
      lerp.inPlace(gradients, tmp, n / Real(noSamples))
    })
    gradients
  }

  final def deriveGradients(predictions: DenseVector[PredictionEx],
                            errors:      DenseVector[Tensor],
                            onEnter:     OnEnterDeriveInputError,
                            onLeave:     OnLeaveDeriveInputError)
  : ParameterBuffer = {
    val layout    = weightsLayout
    val gradients = ParameterBuffer.zeros(layout)
    var noSamples = 0

    predictions.fastForeachEx(errors)((p, e) => {
      val tmp = deriveGradients(p, e, onEnter, onLeave)
      val n = p.input.noSamples
      noSamples += n
      lerp.inPlace(gradients, tmp, n / Real(noSamples))
    })
    gradients
  }

  final def deriveGradients(predictions: DenseMatrix[PredictionEx],
                            errors:      DenseMatrix[Tensor],
                            onEnter:     OnEnterDeriveInputError,
                            onLeave:     OnLeaveDeriveInputError)
  : ParameterBuffer = {
    val layout    = weightsLayout
    val gradients = ParameterBuffer.zeros(layout)
    var noSamples = 0

    predictions.fastForeachEx(errors)((p, e) => {
      val tmp = deriveGradients(p, e, onEnter, onLeave)
      val n = p.input.noSamples
      noSamples += n
      lerp.inPlace(gradients, tmp, n / Real(noSamples))
    })
    gradients
  }
  */


  // ---------------------------------------------------------------------------
  //    Derive gradient (with given input error, with sink)
  // ---------------------------------------------------------------------------
  final def deriveGradients(context: BackpropagationContext,
                            sink:    ValueTensorBuffer)
  : NextError = deriveGradients(
    context,
    sink,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveGradients(context: BackpropagationContext,
                            sink:    ValueTensorBuffer,
                            onEnter: OnEnterDeriveGradients,
                            onLeave: OnLeaveDeriveGradients)
  : NextError = deriveGradients(
    context,
    IndependentError(context.output.createSiblingAndClear()),
    sink,
    onEnter,
    onLeave
  )

  final def deriveGradients(context: BackpropagationContext,
                            error:   Tensor,
                            sink:    ValueTensorBuffer)
  : NextError = deriveGradients(
    context,
    error,
    sink,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveGradients(context: BackpropagationContext,
                            error:   Tensor,
                            sink:    ValueTensorBuffer,
                            onEnter: OnEnterDeriveGradients,
                            onLeave: OnLeaveDeriveGradients)
  : NextError = deriveGradients(
    context,
    IndependentError(error.copy),
    sink,
    onEnter,
    onLeave
  )

  final def deriveGradients(context: BackpropagationContext,
                            error:   NextError,
                            sink:    ValueTensorBuffer,
                            onEnter: OnEnterDeriveGradients,
                            onLeave: OnLeaveDeriveGradients)
  : NextError = {
    // Well, first we should make sure whether this backpropagation context
    // supports backprop at all.
    if (!context.mode.supportsBackpropagation) {
      throw new UnsupportedOperationException("The mode selected during the prediction that produced the backpropagation context does not support backpropgation!")
    }

    // Now push everything we have into a stack so that we can process things
    // one by one.
    val tensors  = mutable.Stack.concat(context.intermediates)
    val contexts = mutable.Stack.concat(context.contexts)
    deriveGradientsEx(error, sink, onEnter, onLeave, tensors, contexts)
  }

  final def deriveGradientsEx(error:         NextError,
                              sink:          ValueTensorBuffer,
                              onEnter:       OnEnterDeriveGradients,
                              onLeave:       OnLeaveDeriveGradients,
                              intermediates: mutable.Stack[Tensor],
                              contexts:      mutable.Stack[PredictContext])
  : NextError = {
    val clock = if (logger.isTraceEnabled) Stopwatch() else null

    val context   = contexts.pop()
    val output    = intermediates.pop()
    val reference = intermediates.pop()
    val input     = intermediates.pop()

    onEnter(this, input, reference, output, context, error)

    val nextError = doDeriveGradients(
      input, reference, output, context,
      error,
      sink,
      onEnter, onLeave,
      intermediates, contexts
    )

    onLeave(this, input, reference, output, context, nextError)

    if (clock != null) {
      logger.trace(f"$clock%s => deriveGradients() => $this%s")
    }
    nextError
  }


  /**
    * Compute actual gradients towards selected weights bank and then (lazy)
    * towards the input. Most layers split the operations.
    *
    * @param intermediates [in] List of layer outputs. Generated from predict.
    * @param reference [in] Reference value used for cost functions.
    * @param sink [in] Buffer for storing the gradients.
    * @param onEnter [in] A function that returns the output errors.
    * @return input error and remaining outputs.
   */
  // THIS HAS JUST A LOCAL SCOPE!
  protected def doDeriveGradients(input:         Tensor,
                                  reference:     Tensor,
                                  output:        Tensor,
                                  context:       PredictContext,
                                  error:         NextError,
                                  sink:          ValueTensorBuffer,
                                  onEnter:       OnEnterDeriveGradients,
                                  onLeave:       OnLeaveDeriveGradients,
                                  intermediates: mutable.Stack[Tensor],
                                  contexts:      mutable.Stack[PredictContext])
  : NextError

  final def deriveGradients(contexts: TraversableOnce[BackpropagationContext],
                            sink:     ValueTensorBuffer)
  : Unit = deriveGradients(
    contexts,
    sink,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveGradients(contexts: TraversableOnce[BackpropagationContext],
                            sink:     ValueTensorBuffer,
                            onEnter:  OnEnterDeriveGradients,
                            onLeave:  OnEnterDeriveGradients)
  : Unit = {
    var n = 0
    contexts.foreach(context => {
      val nextError = deriveGradients(context, sink, onEnter, onLeave)
      nextError.close()
      n += 1
    })
    sink *= Real.one / n
  }

  final def deriveGradients(contexts: Iterator[BackpropagationContext],
                            sink:     ValueTensorBuffer)
  : Unit = deriveGradients(
    contexts,
    sink,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveGradients(contexts: Iterator[BackpropagationContext],
                            sink:     ValueTensorBuffer,
                            onEnter:  OnEnterDeriveGradients,
                            onLeave:  OnEnterDeriveGradients)
  : Unit = {
    var n = 0
    contexts.foreach(context => {
      val nextError = deriveGradients(context, sink, onEnter, onLeave)
      nextError.close()
      n += 1
    })
    sink *= Real.one / n
  }

  final def deriveGradients(contexts: Array[BackpropagationContext],
                            sink:     ValueTensorBuffer)
  : Unit = deriveGradients(
    contexts,
    sink,
    idleOnEnterDeriveInputError,
    idleOnLeaveDeriveInputError
  )

  final def deriveGradients(contexts: Array[BackpropagationContext],
                            sink:     ValueTensorBuffer,
                            onEnter:  OnEnterDeriveGradients,
                            onLeave:  OnEnterDeriveGradients)
  : Unit = {
    ArrayEx.foreach(contexts)(context => {
      val nextError = deriveGradients(context, sink, onEnter, onLeave)
      nextError.close()
    })
    sink *= Real.one / contexts.length
  }



  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : ModuleState = ModuleStateEx(super.state)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: ModuleStateEx =>
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class ModuleBuilder
  extends InstanceExBuilder1[ModuleBuilder, Module, BuildHints]
    with VariantBuilder {

  final private var _handle
  : String = id.toString

  final def handle
  : String = _handle

  final def handle_=(value: String)
  : Unit = {
    require(value != null)
    _handle = value
  }

  def setHandle(value: String)
  : ModuleBuilder

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _handle.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ModuleBuilder =>
      _handle == other._handle
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: ModuleBuilder =>
        other._handle = _handle
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Statistics
  // ---------------------------------------------------------------------------
  final def weightLayoutFor(hints: BuildHints)
  : TensorLayoutBuffer = {
    val builder = TensorLayoutBufferBuilder()
    weightLayoutFor(hints, builder)
    builder.result()
  }

  def weightLayoutFor(hints:   BuildHints,
                      builder: TensorLayoutBufferBuilder)
  : BuildHints

  /**
    * @param hints the input size to try.
    * @return Returns the size of the output of this module, given the input
    *         size.
    */
  def outputHintsFor(hints: BuildHints): BuildHints

  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  final override def build(hints: BuildHints,
                           seed:  InstanceSeed)
  : Module = {
    val weightsBuilder = ValueTensorBufferBuilder()
    build(hints, seed, weightsBuilder)
  }

  def build(hints:          BuildHints,
            seed:           InstanceSeed,
            weightsBuilder: ValueTensorBufferBuilder)
  : Module


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  def permuteHandles(fn: String => String)
  : ModuleBuilder

  protected def doPermuteHandles(fn: String => String)
  : Unit = handle_=(fn(_handle))

  def permuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : ModuleBuilder

  protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {}


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  final def toGraph(hints: BuildHints)
  : Graph = toGraph(Option(hints))

  final def toGraph(hints: Option[BuildHints] = None)
  : Graph = {
    val result = Graph()
    toGraphEx(
      hints,
      Seq.empty,
      LineStyle.Solid,
      result.nodes,
      result.edges
    )
    result
  }

  /**
    * @param nodeSink Vertices and vertex groups will end up here.
    * @param edgeSink Edge information ends up here.
    * @return The vertex for the current object.
    */
  def toGraphEx(hints:     Option[BuildHints],
                inputs:    Seq[Vertex],
                edgeStyle: LineStyle,
                nodeSink:  mutable.Buffer[Node],
                edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex])


  // ---------------------------------------------------------------------------
  //    Checking related
  // ---------------------------------------------------------------------------
  final def check(hints:        BuildHints,
                  indentLevel:  Int    = 0,
                  indentString: String = "  ")
  : Long = checkEx(hints, indentLevel, indentString)._2

  final def checkEx(hints:        BuildHints,
                    indentLevel:  Int    = 0,
                    indentString: String = "  ")
  : (BuildHints, Long) = {
    // Print module header.
    val sb = StringBuilder.newBuilder
    cfor(0)(_ < indentLevel, _ + 1)(
      i => sb ++= indentString
    )
    sb ++= f"$this {"
    logger.info(sb.result())

    // Perform implementation specific tests.
    var noErrors    = 0L
    var outputHints = hints
    try {
      val tmp = doCheckEx(hints, indentLevel, indentString)
      outputHints = tmp._1
      noErrors += tmp._2
    }
    catch {
      case ex: Exception =>
        sb.clear()
        cfor(0)(_ <= indentLevel, _ + 1)(
          i => sb ++= indentString
        )
        sb ++= s"The component does not support supplied hints '$hints'!"
        logger.info(sb.result())
        noErrors += 1L
    }

    // Print number of errors.
    sb.clear()
    cfor(0)(_ <= indentLevel, _ + 1)(
      i => sb ++= indentString
    )
    if (noErrors > 0L) {
      sb ++= f"Number of errors: $noErrors%d"
    }
    else {
      sb ++= "OK"
    }
    logger.info(sb.result())

    // Evaluate and exit.
    sb.clear()
    cfor(0)(_ < indentLevel, _ + 1)(
      i => sb ++= indentString
    )
    sb ++= f"}"
    logger.info(sb.result())
    (outputHints, noErrors)
  }

  protected def doCheckEx(hints:         BuildHints,
                          indentLevel:   Int,
                          indentString:  String)
  : (BuildHints, Long)

}

abstract class ModuleEx[TBuilder <: ModuleExBuilder[_]]
  extends Module {

  override def builder
  : TBuilder

}

abstract class ModuleExBuilder[TThis <: ModuleExBuilder[_]]
  extends ModuleBuilder
    with VariantBuilderEx[TThis] {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

  final override def setHandle(value: String)
  : TThis = {
    handle_=(value)
    repr
  }


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  final override def permuteHandles(fn: String => String)
  : TThis = {
    doPermuteHandles(fn)
    repr
  }

  final override def permuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : ModuleBuilder = {
    doPermuteWeightReferences(fn)
    repr
  }

}

abstract class ModuleState
  extends InstanceState

final case class ModuleStateEx(override val parent: InstanceState)
  extends ModuleState

abstract class ModuleVariantDescription[TBuilder <: ModuleExBuilder[_]]
  extends VariantDescription[TBuilder] {

  final def score(builder:  TBuilder,
                  hints:    BuildHints,
                  priority: Byte)
  : (Int, Array[String]) = {
    val reasons = Array.newBuilder[String]
    var result  = baseScore(builder, priority, reasons)

    // Platform
    if (hints.preferredPlatform.exists(_ == platform)) {
      result |= 1 << 24
      reasons += "platform preference from hints"
    }

    // Avoid tensor format switching.
    if (platform.exists(_ == hints.platform)) {
      result |= 1 << 15
      reasons += "input platform matches"
    }

    // Score overrides.
    result = doScore(builder, hints, result, reasons)
    (result, reasons.result())
  }

  protected def doScore(builder:   TBuilder,
                        hints:     BuildHints,
                        scorePrev: Int,
                        reasons:   mutable.ArrayBuilder[String])
  : Int = scorePrev

  def outputPlatformFor(builder: TBuilder, hints: BuildHints)
  : Platform

  def build(builder:        TBuilder,
            hints:          BuildHints,
            seed:           InstanceSeed,
            weightsBuilder: ValueTensorBufferBuilder)
  : Module

}

class ModuleVariantTable[TBuilder <: ModuleExBuilder[_]]
  extends VariantTable[TBuilder, ModuleVariantDescription[TBuilder]] {

  final def lookup(builder: TBuilder, hints: BuildHints)
  : ModuleVariantDescription[TBuilder] = {
    // Score the variants and select variant with highest score.
    var highestScore: Int = 0
    var highestDesc: ModuleVariantDescription[TBuilder] = null
    MapEx.foreach(variants)((desc, priority) => {
      val (score, reasons) = desc.score(builder, hints, priority)
      if (logger.isDebugEnabled) {
        val sb = StringBuilder.newBuilder
        ArrayEx.foreach(reasons)(reason => {
          sb ++= reason
          sb ++= ", "
        })
        sb.length = Math.max(sb.length - 2, 0)
        logger.debug(f"$builder%s: $score%08x => $desc%s, $sb%s")
      }
      if (score > highestScore) {
        highestScore = score
        highestDesc  = desc
      }
    })

    if (highestDesc == null) {
      throw new UnsupportedOperationException("Unable to determine a compatible variant!")
    }
    if (logger.isInfoEnabled) {
      logger.info(f"$builder%s: $highestDesc%s selected!")
    }
    highestDesc
  }

  final def lookupAndBuild(builder:        TBuilder,
                           hints:          BuildHints,
                           seed:           InstanceSeed,
                           weightsBuilder: ValueTensorBufferBuilder)
  : Module = {
    // Score the the variants.
    val desc = lookup(builder, hints)

    // Instantiate highest and return.
    desc.build(builder, hints, seed, weightsBuilder)
  }

  final def outputPlatformFor(builder: TBuilder, hints: BuildHints)
  : Platform = {
    // Score the the variants.
    val desc = lookup(builder, hints)

    // Instantiate highest and return.
    desc.outputPlatformFor(builder, hints)
  }

}
