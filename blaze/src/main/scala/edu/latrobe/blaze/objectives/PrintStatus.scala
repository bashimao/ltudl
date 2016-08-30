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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import java.io.OutputStreamWriter
import scala.collection._
import scala.util.hashing._

/**
 * A pseudo target that does nothing except printing the progress figures to
 * the console.
 */
final class PrintStatus(override val builder: PrintStatusBuilder,
                        override val seed:    InstanceSeed)
  extends Print[PrintStatusBuilder] {

  private val validators
  : Array[(Validator, ValidationScore => String)] = {
    val builder = Array.newBuilder[(Validator, ValidationScore => String)]
    this.builder.validators.foreach(kv => {
      builder += Tuple2(kv._1.build(seed), kv._2)
    })
    builder.result()
  }

  private var prevIterationTime
  : Timestamp = _

  private var prevValue
  : Real = Real.nan

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : String = {
    val now            = Timestamp.now()
    val iterationNo    = optimizer.iterationNo
    val runNo          = optimizer.runNo
    val runIterationNo = iterationNo - runBeginIterationNo

    // Compute times.
    if (prevIterationTime == null) {
      prevIterationTime = now
    }
    val totalTime = TimeSpan(optimizer.beginTime, now)
    val runTime   = TimeSpan(runBeginTime,        now)
    val diffTime  = TimeSpan(prevIterationTime,   now)

    // Compute cost differences.
    if (Real.isNaN(prevValue)) {
      prevValue = value
    }
    val valueDiff = value - prevValue

    // Sample count.
    val noSamples = {
      if (output != null) {
        output.layout.noSamples
      }
      else {
        0L
      }
    }

    // Assemble debug string.
    val builder = StringBuilder.newBuilder

    builder ++= f"Thr#: ${Thread.currentThread().getId}%3d | "
    builder ++= f"Val: $value%.4g [$valueDiff%+.4e"
    ArrayEx.foreach(validators)(kv => {
      builder ++= ", "
      val validator = kv._1
      val score     = validator(batch.output, output)
      val convFn    = kv._2
      builder ++= convFn(score)
    })
    builder ++= "], "
    builder ++= f"Itr#: $runIterationNo%4d [${diffTime.seconds}%5.3f s] @ ${StringEx.render(noSamples)     }%s | "
    builder ++= f"Run#: $runNo%2d [${          runTime.seconds}%6.1f s] @ ${StringEx.render(runNoSamples)  }%s | "
    builder ++= f"Tot#: $iterationNo%6d [${  totalTime.seconds}%8.1f s]"

    // Update values that keep track of data during previous iteration.
    prevValue         = value
    prevIterationTime = now

    builder.result()
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : ObjectiveState = PrintStatusState(
    super.state,
    ArrayEx.map(
      validators
    )(_._1.state),
    prevIterationTime,
    prevValue
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: PrintStatusState =>
        ArrayEx.foreach(
          validators,
          state.validators
        )(_._1.restoreState(_))
        prevIterationTime = state.prevIterationTime
        prevValue         = state.prevValue
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class PrintStatusBuilder
  extends PrintBuilder[PrintStatusBuilder] {

  override def repr
  : PrintStatusBuilder = this

  val validators
  : mutable.Buffer[(ValidatorBuilder, ValidationScore => String)] = {
    mutable.Buffer.empty
  }

  def +=(validator: (ValidatorBuilder, ValidationScore => String))
  : PrintStatusBuilder = {
    require(validator != null)
    validators += validator
    repr
  }

  def ++=(validators: TraversableOnce[(ValidatorBuilder, ValidationScore => String)])
  : PrintStatusBuilder = {
    validators.foreach(
      this += _
    )
    repr
  }

  override protected def doToString()
  : List[Any] = validators.length :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), validators.hashCode)

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PrintStatusBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PrintStatusBuilder =>
      validators == other.validators
    case _ =>
      false
  })

  override protected def doCopy()
  : PrintStatusBuilder = PrintStatusBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PrintStatusBuilder =>
        other.validators.clear()
        other.validators ++= validators.map(
          kv => Tuple2(kv._1.copy, kv._2)
        )
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : PrintStatus = new PrintStatus(this, seed)

}

object PrintStatusBuilder {

  final def apply()
  : PrintStatusBuilder = new PrintStatusBuilder

  final def apply(validator0: (ValidatorBuilder, ValidationScore => String))
  : PrintStatusBuilder = apply() += validator0

  final def apply(validators: TraversableOnce[(ValidatorBuilder, ValidationScore => String)])
  : PrintStatusBuilder = apply() ++= validators

}

final case class PrintStatusState(override val parent: InstanceState,
                                  validators:          Array[InstanceState],
                                  prevIterationTime:   Timestamp,
                                  prevValue:           Real)
  extends ObjectiveState {
}
