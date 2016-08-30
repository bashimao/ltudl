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

/**
  * If you close the companion prediction backprop is no longer feasible.
  *
  * @param contexts A object that contains information for back-propagation. Can
  *                 safely be ditched if not required. May need to call
  *                 drop if using modules that do native memory allocations.
  */
final class BackpropagationContext(val mode:          Mode,
                                   val input:         Tensor,
                                   val reference:     Tensor,
                                   val output:        Tensor,
                                   var value:         Real,
                                   val intermediates: List[Tensor],
                                   val contexts:      List[PredictContext])
  extends AutoCloseable {

  private var handedOffOutput
  : Boolean = false

  def cost
  : Cost = Cost(value, output.layout.noSamples)

  /**
    * Drops backprop related data.
    */
  override def close()
  : Unit = {
    intermediates.foreach(tensor => {
      if (tensor != null) {
        if (tensor ne input) {
          if (tensor ne reference) {
            if (tensor ne output) {
              tensor.tryClose()
            }
          }
        }
      }
    })
    contexts.foreach(_.close())
    if (!handedOffOutput) {
      output.close()
    }
  }

  def dropIntermediates()
  : Prediction = {
    handedOffOutput = true
    close()
    Prediction(input, reference, output, value)
  }

}

object BackpropagationContext {

  final def newBuilder(input: Tensor, reference: Tensor)
  : BackpropagationContextBuilder = new BackpropagationContextBuilder(
    input,
    reference
  )

}

final class BackpropagationContextBuilder(private val input:     Tensor,
                                          private val reference: Tensor) {

  private var blockedTensors: List[Tensor] = Nil

  private var actuallyBlockedTensors: List[Tensor] = Nil

  private var intermediates: List[Tensor] = Nil

  private var actualIntermediates: List[Tensor] = Nil

  private var contexts: List[PredictContext] = Nil

  block(input)
  block(reference)

  def block(tensor: Tensor)
  : Unit = {
    blockedTensors = tensor :: blockedTensors
    blockRecursive(tensor)
  }

  private def blockRecursive(tensor: Tensor)
  : Unit = {
    actuallyBlockedTensors = tensor :: actuallyBlockedTensors
    tensor match {
      case tensor: TensorTable =>
        tensor.foreachTensor(blockRecursive(_))
      case _ =>
    }
  }

  def unblock(): Unit = {
    unblockRecursive(blockedTensors.head)
    blockedTensors = blockedTensors.tail
  }

  private def unblockRecursive(tensor: Tensor)
  : Unit = {
    actuallyBlockedTensors = actuallyBlockedTensors.tail
    tensor match {
      case tensor: TensorTable =>
        tensor.foreachTensor(unblockRecursive(_))
      case _ =>
    }
  }

  def isBlocked(tensor: Tensor)
  : Boolean = tensor match {
    case tensor: TensorTable =>
      val iter = tensor.iterator
      while (iter.hasNext) {
        if (isBlocked(iter.next())) {
          return true
        }
      }
      false
    case _ =>
      actuallyBlockedTensors.exists(_ eq tensor)
  }

  def stash(tensor: Tensor)
  : Unit = {
    intermediates = tensor :: intermediates
    stashRecursive(tensor)
  }

  private def stashRecursive(tensor: Tensor)
  : Unit = {
    if (tensor != null) {
      if (!actualIntermediates.exists(_ eq tensor)) {
        actualIntermediates = tensor :: actualIntermediates
        tensor match {
          case tensor: TensorTable =>
            tensor.foreachTensor(stashRecursive(_))
          case _ =>
        }
      }
    }
  }

  def stash(context: PredictContext)
  : Unit = contexts = context :: contexts

  def stashSize
  : Map[Platform, Long] = {
    val result = mutable.HashMap.empty[Platform, Long]
    actualIntermediates.foreach(tensor => {
      val platform = tensor.platform
      val oldSize  = result.getOrElseUpdate(platform, 0L)
      val newSize  = tensor.layout.noValues + oldSize
      result += Tuple2(tensor.platform, newSize)
    })
    result
  }

  def requiresMaintaining(tensor: Tensor)
  : Boolean = tensor match {
    case tensor: TensorTable =>
      val iter = tensor.iterator
      while (iter.hasNext) {
        if (requiresMaintaining(iter.next())) {
          return true
        }
      }
      false
    case _ =>
      actualIntermediates.exists(_ eq tensor)
  }

  def result(mode: Mode, output: Tensor, cost: Real)
  : BackpropagationContext = new BackpropagationContext(
    mode,
    input,
    reference,
    output,
    cost,
    intermediates,
    contexts
  )

}