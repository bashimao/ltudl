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
import edu.latrobe.io.graph._
import scala.Seq
import scala.collection._
import scala.util.hashing._

final class AutoEncoder(override val builder:             AutoEncoderBuilder,
                        override val inputHints:          BuildHints,
                        override val seed:                InstanceSeed,
                        override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Container[AutoEncoderBuilder] {

  val encoder
  : Module = builder.encoder.build(
    inputHints, seed, weightBufferBuilder
  )

  override val outputHints
  : BuildHints = encoder.outputHints

  val intermediateInputHints
  : BuildHints = outputHints.derive(
    outputHints.platform, outputHints.layout,
    inputHints.platform,  inputHints.layout
  )

  val decoder
  : Module = builder.decoder.build(
    intermediateInputHints, seed, weightBufferBuilder
  )

  val intermediateOutputHints
  : BuildHints = decoder.outputHints

  val terminator
  : Module = builder.terminator.build(
    intermediateOutputHints, seed, weightBufferBuilder
  )

  override val children
  : Seq[Module] = Seq(encoder, decoder, terminator)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override val requiresMaintainingInputDuringForwardPropagation
  : Boolean = true

  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor,
                                   onEnter:        OnEnterPredict,
                                   onLeave:        OnLeavePredict)
  : (Tensor, PredictContext) = {
    // Encoder part.
    val encOut = encoder.predictEx(
      mode,
      input,
      reference,
      onEnter,
      onLeave
    )

    // Run through decoder.
    val decOut = decoder.predictEx(
      mode,
      encOut,
      input,
      onEnter,
      onLeave
    )

    // Run decoder part through branch terminator.
    terminator.predictEx(
      mode,
      decOut,
      reference,
      onEnter,
      onLeave
    )

    // Hand over encoder output.
    (encOut, EmptyContext)
  }

  override protected def doPredictInv(output:   Tensor,
                                      context:  PredictContext,
                                      onLeave:  OnLeavePredict,
                                      contexts: mutable.Stack[PredictContext])
  : Tensor = encoder.predictInvEx(output, onLeave, contexts)


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveGradients(context:       PredictContext,
                                           error:         NextError,
                                           sink:          ValueTensorBuffer,
                                           onEnter:       OnEnterDeriveGradients,
                                           onLeave:       OnLeaveDeriveGradients,
                                           intermediates: mutable.Stack[Tensor],
                                           contexts:      mutable.Stack[PredictContext])
  : NextError = {
    val err0 = error.compute()

    // Run through branch terminator.
    var error1 = terminator.deriveGradientsEx(
      null,
      sink,
      onEnter,
      onLeave,
      intermediates,
      contexts
    )

    // Decoder.
    error1 = decoder.deriveGradientsEx(
      error1,
      sink,
      onEnter,
      onLeave,
      intermediates,
      contexts
    )

    // Add decoder error on top.
    using(
      error1.compute()
    )(err0 += _)

    // Run through encoder
    encoder.deriveGradientsEx(
      IndependentError.derive(err0),
      sink,
      onEnter,
      onLeave,
      intermediates,
      contexts
    )
  }


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : AutoEncoderState = AutoEncoderState(
    super.state,
    encoder.state,
    decoder.state
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: AutoEncoderState =>
        encoder.restoreState(state.encoder)
        decoder.restoreState(state.decoder)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class AutoEncoderBuilder
  extends ContainerBuilder[AutoEncoderBuilder] {

  override def repr
  : AutoEncoderBuilder = this

  private var _encoder
  : ModuleBuilder = IdentityBuilder()

  def encoder
  : ModuleBuilder = _encoder

  def encoder_=(value: ModuleBuilder)
  : Unit = {
    require(value != null)
    _encoder = value
  }

  def setEncoder(value: ModuleBuilder)
  : AutoEncoderBuilder = {
    encoder_=(value)
    this
  }

  private var _decoder
  : ModuleBuilder = IdentityBuilder()

  def decoder
  : ModuleBuilder = _decoder

  def decoder_=(value: ModuleBuilder)
  : Unit = {
    require(value != null)
    _decoder = value
  }

  def setDecoder(value: ModuleBuilder)
  : AutoEncoderBuilder = {
    decoder_=(value)
    this
  }

  private var _terminator
  : ModuleBuilder = BranchTerminatorBuilder()

  def terminator
  : ModuleBuilder = _terminator

  def terminator_=(value: ModuleBuilder)
  : Unit = {
    require(value != null)
    _terminator = value
  }

  def setTerminator(value: ModuleBuilder)
  : AutoEncoderBuilder = {
    terminator_=(value)
    this
  }

  override def children
  : Seq[ModuleBuilder] = Seq(
    _encoder,
    _decoder,
    _terminator
  )

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AutoEncoderBuilder]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _encoder.hashCode())
    tmp = MurmurHash3.mix(tmp, _decoder.hashCode())
    tmp = MurmurHash3.mix(tmp, _terminator.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AutoEncoderBuilder =>
      _encoder    == other._encoder &&
      _decoder    == other._decoder &&
      _terminator == other._terminator
    case _ =>
      false
  })

  override protected def doCopy()
  : AutoEncoderBuilder = AutoEncoderBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AutoEncoderBuilder =>
        other._encoder    = _encoder.copy
        other._decoder    = _decoder.copy
        other._terminator = _terminator.copy
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Statistics
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    val encHints = _encoder.weightLayoutFor(hints, builder)
    val tmpHints = encHints.derive(
      encHints.platform, encHints.layout,
      hints.platform,    hints.layout
    )
    _decoder.weightLayoutFor(tmpHints, builder)
    encHints
  }

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = _encoder.outputHintsFor(hints)


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def build(hints:                BuildHints,
                     seed:                 InstanceSeed,
                     weightsBufferBuilder: ValueTensorBufferBuilder)
  : AutoEncoder = new AutoEncoder(this, hints, seed, weightsBufferBuilder)


  // ---------------------------------------------------------------------------
  //    Checking related
  // ---------------------------------------------------------------------------
  override protected def doCheckEx(hints:        BuildHints,
                                   indentLevel:  Int,
                                   indentString: String)
  : (BuildHints, Long) = {
    val (encHints, encErrors) = _encoder.checkEx(
      hints,
      indentLevel + 1,
      indentString
    )

    val tmpHints = encHints.derive(
      encHints.platform, encHints.layout,
      hints.platform,    hints.layout
    )
    val decErrors = _decoder.check(tmpHints, indentLevel + 1, indentString)

    (encHints, encErrors + decErrors)
  }


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToGraphEx(node:      VertexGroup,
                                     hints:     Option[BuildHints],
                                     inputs:    Seq[Vertex],
                                     edgeStyle: LineStyle,
                                     nodeSink:  mutable.Buffer[Node],
                                     edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Push everything through the encoder.
    val (encHints, encOutputs) = _encoder.toGraphEx(
      hints,
      inputs,
      edgeStyle,
      nodeSink,
      edgeSink
    )

    // Compute decoder input.
    val tmpHints = encHints.map(encHints => {
      val _hints = hints.get
      encHints.derive(
        encHints.platform, encHints.layout,
        _hints.platform,   _hints.layout
      )
    })

    // Run through decoder.
    val (decHints, decOutputs) = _decoder.toGraphEx(
      tmpHints,
      encOutputs ++ inputs,
      LineStyle.Solid,
      nodeSink,
      edgeSink
    )

    // Run signals through terminator.
    _terminator.toGraphEx(
      decHints,
      decOutputs,
      LineStyle.Solid,
      nodeSink,
      edgeSink
    )

    // But only present the encoder's output to the next module.
    (encHints, encOutputs)
  }

}

object AutoEncoderBuilder {

  final def apply()
  : AutoEncoderBuilder = new AutoEncoderBuilder

  final def apply(encoder: ModuleBuilder,
                  decoder: ModuleBuilder)
  : AutoEncoderBuilder = apply().setEncoder(encoder).setDecoder(decoder)

}

final case class AutoEncoderState(override val parent: InstanceState,
                                  encoder:             InstanceState,
                                  decoder:             InstanceState)
  extends ModuleState {
}

final case class AutoEncoderContext(encoder:           PredictContext,
                                    decoder:           PredictContext)
  extends PredictContext {

  override protected def doClose()
  : Unit = {
    decoder.close()
    encoder.close()
    super.doClose()
  }

}
