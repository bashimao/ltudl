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

package edu.latrobe.blaze.optimizers

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.optimizerexitcodes._
import edu.latrobe.time._
import java.util.UUID
import scala.collection._
import scala.util.hashing._

/**
  * Scala adaptation of the conjugate gradient decent matlab function
  * originally developed by Carl Edward Rasmussen.
  *
  * http://www.gatsby.ucl.ac.uk/~edward/code/minimize
  *
  * I rearranged a few bits and pieces, so that it fits our model a more.
  */
final class ConjugateGradientDescent(override val builder:   ConjugateGradientDescentBuilder,
                                     override val model:     Module,
                                     override val batchPool: BatchPool,
                                     override val seed:      InstanceSeed)
  extends OptimizerEx[ConjugateGradientDescentBuilder] {

  val rho: Real = builder.rho

  val sig: Real = builder.sig

  val int: Real = builder.int

  val ext: Real = builder.ext

  val max: Int = builder.max

  val ratio: Real = builder.ratio

  override val buffers
  : List[ValueTensorBuffer] = super.buffers

  override val parameters
  : Map[UUID, Parameter] = super.parameters

  // TODO: This does not really cleanup tensors correctly. Relies on GC!
  override protected def doRun(runBeginIterationNo: Long,
                               runBeginTime:        Timestamp)
  : OptimizationResult = {
    var noSamples = 0L

    // no previous line search has failed
    var noFailures = 0

    // M_LANGER: Draw next batch from the pool.
    // TODO: This is actually inaccurate and will not work properly if we have multiple banks!
    // TODO: Add in scope support!
    //var scope       = this.scope.get(_iterationNo, runNo)
    var drawContext = batchPool.draw()
    if (drawContext.isEmpty) {
      return OptimizationResult.derive(NoMoreData(), _iterationNo, noSamples)
    }
    var batch = drawContext.batch

    // get function value and gradient
    using(
      weightBuffer.allocateSibling(),
      weightBuffer.allocateSibling()
    )((_df1, _df2) => {
      var df = (_df1, _df2)
      var f1 = doForwardAndBackwardProp(weightBuffer, batch, df._1)

      // search direction is steepest
      using(
        -df._1,
        weightBuffer.allocateSibling(),
        weightBuffer.allocateSibling()
      )((s, X0, df0) => {
        // this is the slope
        var d1 = df._1.dot(s)
        // initial step is red/(|s|+1)
        var z1 = Real.one / (Real.one - d1)

        // while not finished
        while (_iterationNo < Long.MaxValue) {

          // make a copy of current values
          X0 := weightBuffer
          val f0 = f1
          df0 := df._1

          // begin line search
          weightBuffer.add(s, z1)
          var f2 = doForwardAndBackwardProp(weightBuffer, batch, df._2)
          var d2 = df._2.dot(s)

          // initialize point 3 equal to point 1
          var f3 = f1
          var d3 = d1
          var z3 = -z1

          // initialize quantities
          var M = max

          var limit = -Real.one

          def lineSearch(): Boolean = {
            while (true) {
              while (
                (M > 0) &&
                ((f2 > f1 + z1 * rho * d1) || (d2 > -sig * d1))
              ) {
                // tighten the bracket
                limit = z1
                var z2 = Real(
                  if (f2 > f1) {
                    // quadratic fit
                    z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)
                  }
                  else {
                    // cubic fit
                    val A = (f2 - f3) * 6.0 / z3 + (d3 + d2) * 3.0
                    val B = (f3 - f2) * 3.0 - z3 * (d3 + d2 + d2)

                    // numerical error possible -ok !
                    (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A
                  }
                )

                // if we had a numerical problem then bisect
                if (Real.isNaN(z2) || Real.isInfinite(z2)) {
                  z2 = z3 * Real.pointFive
                }

                // don't accept too close to limits
                z2 = Math.max(Math.min(z2, int * z3), (Real.one - int) * z3)

                // update the step
                z1 += z2
                weightBuffer.add(s, z2)
                f2 = doForwardAndBackwardProp(weightBuffer, batch, df._2)
                d2 = df._2.dot(s)

                // z3 is now relative to the location of z2
                z3 -= z2

                M -= 1
              }

              // this is a failure
              if (f2 > f1 + z1 * rho * d1 || d2 > -sig * d1) {
                logger.warn(s"Line search failed! (E=0 M=$M)")
                return false
              }
              // success
              else if (d2 > sig * d1) {
                logger.trace(s"Line search success! (M=$M)")
                return true
              }
              // failure
              else if (M == 0) {
                logger.warn(s"Line search failed! (E=1 M=$M)")
                return false
              }

              var z2 = Real({
                // make cubic extrapolation
                val A = (f2 - f3) * 6.0 / z3 + (d3 + d2) * 3.0
                val B = (f3 - f2) * 3.0 - z3 * (d3 + d2 + d2)

                // num. error possible - ok!
                -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3))
              })

              // num prob or wrong sign?
              if (Real.isNaN(z2) || Real.isInfinite(z2) || z2 < Real.zero) {
                z2 = {
                  // if we have no upper limit
                  if (limit < -Real.pointFive) {
                    // the extrapolate the maximum amount
                    z1 * (ext - Real.one)
                  }
                  else {
                    // otherwise bisect
                    (limit - z1) * Real.pointFive
                  }
                }
              }
              // extrapolation beyond max ?
              else if ((limit > -Real.pointFive) && (z2 + z1 > limit)) {
                // bisect
                z2 = (limit - z1) * Real.pointFive
              }
              // extrapolation beyond limit
              else if ((limit < -Real.pointFive) && (z2 + z1 > z1 * ext)) {
                // set to extrapolation limit
                z2 = z1 * (ext - Real.one)
              }
              else if (z2 < -z3 * int) {
                z2 = -z3 * int
              }
              // too close to limit?
              else if (
                (limit > -Real.pointFive) &&
                (z2 < (limit - z1) * (Real.one - int))
              ) {
                z2 = (limit - z1) * (Real.one - int)
              }

              // set point 3 equal to point 2
              f3 = f2
              d3 = d2
              z3 = -z2

              // update current estimates
              z1 += z2
              weightBuffer.add(s, z2)
              f2 = doForwardAndBackwardProp(weightBuffer, batch, df._2)
              d2 = df._2.dot(s)

              M -= 1
            } // end of line search

            false
          }

          // if line search succeeded
          val success = lineSearch()
          if (success) {
            noSamples += batch.noSamples

            // M_LANGER: Evaluate objectives.
            doEvaluateEarlyObjectives(
              runBeginIterationNo,
              runBeginTime,
              noSamples
            ).foreach(exitCode => {
              return OptimizationResult.derive(
                exitCode, _iterationNo, noSamples
              )
            })
            doEvaluateObjectives(
              runBeginIterationNo,
              runBeginTime,
              noSamples,
              batch, batch.output, f2 // TODO: This is actually wrong!
            ).foreach(exitCode => {
              return OptimizationResult.derive(
                exitCode, _iterationNo, noSamples
              )
            })

            // Polack-Ribiere direction
            s *= ((df._2.dot(df._2) - df._1.dot(df._2)) / df._1.dot(df._1))
            s -= df._2
            d2 = df._2.dot(s)

            // new slope must be negative
            if (d2 > Real.zero) {
              // otherwise use steepest direction
              s := df._2
              s *= -Real.one
              d2 = df._2.dot(s)
            }

            // slope ratio but max RATIO
            z1 *= Math.min(ratio, d1 / (d2 - Real.epsilon))

            // swap derivatives
            f1 = f2
            d1 = d2
            df = df.swap

            // this line search did not fail
            noFailures = 0

            // M_LANGER: Let's fetch the next batch.
            drawContext.close()
            drawContext = batchPool.draw()
            if (drawContext.isEmpty) {
              return OptimizationResult.derive(
                NoMoreData(), _iterationNo, noSamples
              )
            }
            batch = drawContext.batch

            // M_LANGER: Update hyper parameters.
            doUpdateParameters(_iterationNo, f1)
            //scope.update(_iterationNo, f1)
            // TODO: Fix scope support!
            //bankNo = scope.get(_iterationNo, runNo)
          }
          else {
            // restore point from before failed line search
            weightBuffer := X0
            f1 = f0
            df._1 := df0

            // line search failed twice in a row
            if (noFailures > 0) {
              return OptimizationResult.derive(
                LineSearchFailed(), _iterationNo, noSamples
              )
            }

            // swap derivatives
            df = df.swap

            // try steepest
            s := df._1
            s *= -Real.one
            d1 = df._1.dot(s)
            z1 = Real.one / (Real.one - d1)

            // this line search failed
            noFailures += 1
          }

          // M_LANGER: Next...
          _iterationNo += 1L
        }
      })
    })

    OptimizationResult.derive(NoIterationsLimit(), _iterationNo, noSamples)
  }

}

final class ConjugateGradientDescentBuilder
  extends OptimizerExBuilder[ConjugateGradientDescentBuilder] {

  override def repr
  : ConjugateGradientDescentBuilder = this

  /**
    * RHO and SIG are the constants in the Wolfe-Powell conditions
    */
  private var _rho
  : Real = 0.01f

  def rho
  : Real = _rho

  def rho_=(value: Real)
  : Unit = {
    require(value > Real.zero)
    _rho = value
  }

  def setRho(value: Real)
  : ConjugateGradientDescentBuilder = {
    rho_=(value)
    this
  }

  /**
    * RHO and SIG are the constants in the Wolfe-Powell conditions
    */
  private var _sig
  : Real = Real.pointFive

  def sig
  : Real = _sig

  def sig_=(value: Real)
  : Unit = {
    require(value > Real.zero)
    _sig = value
  }

  def setSig(value: Real)
  : ConjugateGradientDescentBuilder = {
    sig_=(value)
    this
  }

  /**
    * Don't reevaluate within 0.1 of the limit of the current bracket
    */
  private var _int
  : Real = 0.1f

  def int
  : Real = _int

  def int_=(value: Real)
  : Unit = {
    require(value > Real.zero)
    _int = value
  }

  def setInt(value: Real)
  : ConjugateGradientDescentBuilder = {
    int_=(value)
    this
  }

  /**
    * Extrapolate maximum 3 times the current bracket.
    */
  private var _ext
  : Real = 3.0f

  def ext
  : Real = _ext

  def ext_=(value: Real)
  : Unit = {
    require(value > Real.zero)
    _ext = value
  }

  def setExt(value: Real)
  : ConjugateGradientDescentBuilder = {
    ext_=(value)
    this
  }

  /**
    * Max 20 function evaluations per line search.
    */
  private var _max
  : Int = 20

  def max
  : Int = _max

  def max_=(value: Int)
  : Unit = {
    require(value > 0)
    _max = value
  }

  def setMax(value: Int)
  : ConjugateGradientDescentBuilder = {
    max_=(value)
    this
  }

  /**
    * Maximum allowed slope ratio.
    */
  private var _ratio
  : Real = 100.0f

  def ratio
  : Real = _ratio

  def ratio_=(value: Real)
  : Unit = {
    require(value > Real.zero)
    _ratio = value
  }

  def setRatio(value: Real)
  : ConjugateGradientDescentBuilder = {
    ratio_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    f"${_rho}%.4g" :: f"${_sig}%.4g" :: f"${_int}%.4g" :: f"${_ext}%.4g" :: _max :: f"${_ratio}%.4g" :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _rho.hashCode())
    tmp = MurmurHash3.mix(tmp, _sig.hashCode())
    tmp = MurmurHash3.mix(tmp, _int.hashCode())
    tmp = MurmurHash3.mix(tmp, _ext.hashCode())
    tmp = MurmurHash3.mix(tmp, _max.hashCode())
    tmp = MurmurHash3.mix(tmp, _ratio.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConjugateGradientDescentBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConjugateGradientDescentBuilder =>
      _rho   == other._rho &&
      _sig   == other._sig &&
      _int   == other._int &&
      _ext   == other._ext &&
      _max   == other._max &&
      _ratio == other._ratio
    case _ =>
      false
  })

  override protected def doCopy()
  : ConjugateGradientDescentBuilder = ConjugateGradientDescentBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConjugateGradientDescentBuilder =>
        other._rho   = _rho
        other._sig   = _sig
        other._int   = _int
        other._ext   = _ext
        other._max   = _max
        other._ratio = _ratio
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : ConjugateGradientDescent = new ConjugateGradientDescent(
    this, model, batchPool, seed
  )

}

object ConjugateGradientDescentBuilder {

  final def apply()
  : ConjugateGradientDescentBuilder = new ConjugateGradientDescentBuilder()

}
