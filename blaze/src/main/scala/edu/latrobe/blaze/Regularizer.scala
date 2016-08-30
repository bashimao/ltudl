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
  * Unlike modules, regularizers operate directly on the weights/gradients
  * level. They can - for example - be used to constraint weights from growing
  * too large or too small.
  *
  * Note that regularizers may or may not change the cost of the model.
  */
abstract class Regularizer
  extends InstanceEx[RegularizerBuilder]
    with ParameterizedInstance {

  final val baseScope
  : Option[NullBuffer] = builder.baseScope

  /**
    * Must implement as constructor argument!
    */
  def platformHint
  : Option[Platform]


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  def evaluate(phaseNo:   Long,
               weights:   ValueTensorBuffer,
               input:     Tensor,
               reference: Tensor,
               output:    Tensor)
  : Real

  final def evaluate(phaseNo: Long,
                     weights: ValueTensorBuffer,
                     batch:   Batch)
  : Real = evaluate(
    phaseNo,
    weights,
    batch.input,
    batch.output,
    null
  )

  final def evaluate(phaseNo: Long,
                     weights: ValueTensorBuffer,
                     context: BackpropagationContext)
  : Real = evaluate(
    phaseNo,
    weights,
    context.input,
    context.reference,
    context.output
  )


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  def deriveGradients(phaseNo:   Long,
                      weights:   ValueTensorBuffer,
                      input:     Tensor,
                      reference: Tensor,
                      output:    Tensor,
                      sink:      ValueTensorBuffer)
  : Unit

  final def deriveGradients(phaseNo: Long,
                            weights: ValueTensorBuffer,
                            context: BackpropagationContext,
                            sink:    ValueTensorBuffer)
  : Unit = deriveGradients(
    phaseNo,
    weights,
    context.input,
    context.reference,
    context.output,
    sink
  )


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : RegularizerState = RegularizerStateEx(super.state)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: RegularizerStateEx =>
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class RegularizerBuilder
  extends InstanceExBuilder1[RegularizerBuilder, Regularizer, Option[Platform]]
    with VariantBuilder {

  final var _baseScope
  : Option[NullBuffer] = None

  final def baseScope
  : Option[NullBuffer] = _baseScope

  final def baseScope_=(value: Option[NullBuffer])
  : Unit = {
    require(value != null)
    _baseScope = value
  }

  def setBaseScope(value: Option[NullBuffer])
  : RegularizerBuilder

  def setBaseScope(value: NullBuffer)
  : RegularizerBuilder


  // ---------------------------------------------------------------------------
  //    Object building related.
  // ---------------------------------------------------------------------------
  override def build(platformHint: Option[Platform],
                     seed:         InstanceSeed)
  : Regularizer

  final def build(platformHint: Platform)
  : Regularizer = build(platformHint, InstanceSeed.default)

  final def build(platformHint: Platform,
                  seed:         InstanceSeed)
  : Regularizer = build(Option(platformHint), seed)

}

abstract class RegularizerEx[TBuilder <: RegularizerExBuilder[_]]
  extends Regularizer {

  override def builder
  : TBuilder

}

abstract class RegularizerExBuilder[TThis <: RegularizerExBuilder[_]]
  extends RegularizerBuilder
    with VariantBuilderEx[TThis] {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

  final override def setBaseScope(value: Option[NullBuffer])
  : TThis = {
    baseScope_=(value)
    repr
  }

  final override def setBaseScope(value: NullBuffer)
  : TThis = setBaseScope(Option(value))

}

abstract class RegularizerVariantDescription[TBuilder <: RegularizerExBuilder[TBuilder]]
  extends VariantDescription[TBuilder] {

  final def score(builder:      TBuilder,
                  platformHint: Option[Platform],
                  priority:     Byte)
  : (Int, Array[String]) = {
    val reasons = Array.newBuilder[String]
    var result = baseScore(builder, priority, reasons)

    // Platform
    if (platformHint.exists(_ == platform)) {
      result |= 1 << 24
      reasons += "build level platform preference"
    }

    // Score overrides.
    result = doScore(builder, platformHint, result, reasons)
    (result, reasons.result())
  }

  protected def doScore(builder:      TBuilder,
                        platformHint: Option[Platform],
                        scorePrev:    Int,
                        reasons:      mutable.ArrayBuilder[String])
  : Int = scorePrev

  def build(builder:      TBuilder,
            platformHint: Option[Platform],
            seed:         InstanceSeed)
  : Regularizer

}

class RegularizerVariantTable[TBuilder <: RegularizerExBuilder[TBuilder]]
  extends VariantTable[TBuilder, RegularizerVariantDescription[TBuilder]] {

  final def lookup(builder:      TBuilder,
                   platformHint: Option[Platform])
  : RegularizerVariantDescription[TBuilder] = {
    // Score the variants and select variant with highest score.
    var highestScore: Int = 0
    var highestDesc: RegularizerVariantDescription[TBuilder] = null
    MapEx.foreach(variants)((desc, priority) => {
      val (score, reasons) = desc.score(builder, platformHint, priority)
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

  final def lookupAndBuild(builder:      TBuilder,
                           platformHint: Option[Platform],
                           seed:         InstanceSeed)
  : Regularizer = {
    // Score the the variants.
    val desc = lookup(builder, platformHint)

    // Instantiate highest and return.
    desc.build(builder, platformHint, seed)
  }

}

abstract class RegularizerState
  extends InstanceState

final case class RegularizerStateEx(override val parent: InstanceState)
  extends RegularizerState