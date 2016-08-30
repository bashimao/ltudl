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

package edu.latrobe.blaze.regularizers

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.regularizers.generic._
import edu.latrobe.blaze.regularizers.jvm._
import scala.util.hashing._

/*
final override def computeRawGradientsForFilter(mode:   ComputeMode,
                                                error:  SampleTensor,
                                                input:  SampleTensor,
                                                result: DVec)
: Unit = {
  //val offset2    = offset1 + wFlat.length
  //val gradientsW = result(offset1 until offset2)

  // This avoid creating a huge matrix of unmanageable size which we would
  // anyway not need.
  val res = result.asMatrix(weightsMatrix.rows, weightsMatrix.cols)
  val err = error.values
  val inp = input.values

  kernel.foreachValidPair(outputSize, noMaps, (i0, i1, offset0) => {
    /*val n0   = i  * noMaps
    val nEnd = n0 + noMaps
    */
    val outRange = i0 until i1
    // TODO: Why not use w vectorized?
    //val w0   = n0 * kernel.noValues
    val dstSlice = res(::, outRange)
    val src_t    = inp(outRange).t

    (j0, j1, offset0, offset1) => {
      /*
      val tmp = error.valueAt(offset)
      var w   = w0 + j
      var n   = n0
      while (n < nEnd) {
        // TODO: Why use update?
        //test.data(w0 + m) = a dot rawError(n0 + m, ::)
        resultW.update(w, tmp * in.valueAt(n))
        w += kernel.noValues
        n += 1
      }
      */
      val e   = err(offset0 until offset1)
      val tmp = e * src_t
      val dst = dstSlice(j0 until j1, ::)
      dst += tmp
    }
  })

  /*
  // TODO: Have to find a way to do this better without wasting too much memory.
  // TODO: Just to avoid as single allocation? Isn't that a little bit much effort?
  if (lambda._1.isNaN && lambda._2.isNaN) {
    kernel.foreachPair((i, offset) => {
      val n0   = i  * noMaps
      val nEnd = n0 + noMaps
      // TODO: Why not use w vectorized?
      val w0   = n0 * kernel.size

      (j, offset) => {
        val tmp = rawError.valueAt(offset)
        var w   = w0 + j
        var n   = n0
        while (n < nEnd) {
          // TODO: Why use update?
          //test.data(w0 + m) = a dot rawError(n0 + m, ::)
          gradientsW.update(w, tmp * in.valueAt(n))
          w += kernel.size
          n += 1
        }
      }
    })
  }
  else if (lambda._1.isNaN) {
    gradientsW := wFlat
    gradientsW *= lambda._2
    kernel.foreachPair((i, offset) => {
      val n0   = i  * noMaps
      val nEnd = n0 + noMaps
      // TODO: Why not use w vectorized?
      val w0   = n0 * kernel.size

      (j, offset) => {
        val tmp = rawError.valueAt(offset)
        var w   = w0 + j
        var n   = n0
        while (n < nEnd) {
          gradientsW(w) += tmp * in.valueAt(n)
          w             += kernel.size
          n             += 1
        }
      }
    })
  }
  else if (lambda._2.isNaN) {
    gradientsW := wFlat
    gradientsW *= lambda._1
    val tmp2 = wFlat :* wFlat
    tmp2 += epsilon
    sqrt.inPlace(tmp2)
    gradientsW :/= tmp2
    kernel.foreachPair((i, offset) => {
      val n0   = i * noMaps
      val nEnd = n0 + noMaps
      // TODO: Why not use w vectorized?
      val w0   = n0 * kernel.size

      (j, offset) => {
        val tmp = rawError.valueAt(offset)
        var w   = w0 + j
        var n   = n0
        while (n < nEnd) {
          gradientsW(w) += tmp * in.valueAt(n)
          w             += kernel.size
          n             += 1
        }
      }
    })
  }
  else {
    gradientsW := wFlat
    gradientsW *= lambda._1
    val tmp2 = wFlat :* wFlat
    tmp2 += epsilon
    sqrt.inPlace(tmp2)
    gradientsW :/= tmp2
    gradientsW +=  wFlat * lambda._2
    kernel.foreachPair((i, offset) => {
      val n0   = i  * noMaps
      val nEnd = n0 + noMaps
      // TODO: Why not use w vectorized?
      val w0   = n0 * kernel.size

      (j, offset) => {
        val tmp = rawError.valueAt(offset)
        var w   = w0 + j
        var n   = n0
        while (n < nEnd) {
          gradientsW(w) += tmp * in.valueAt(n)
          w             += kernel.size
          n             += 1
        }
      }
    })
  }
  */
}
*/
//val offset2    = offset1 + wFlat.length
//val gradientsW = result(offset1 until offset2)
//val gradientsW = result(wFlat.offsets)

/*
// Regularization
if (lambda._1.isNaN && lambda._2.isNaN) {
  gradientsW := Real.zero
}
else if (lambda._1.isNaN) {
  gradientsW := wFlat
  gradientsW *= outputSize * rawError.cols * lambda._2
}
else if (lambda._2.isNaN) {
  gradientsW := wFlat
  gradientsW *= outputSize * rawError.cols * lambda._1
  val tmp2 = wFlat :* wFlat
  tmp2 += epsilon
  sqrt.inPlace(tmp2)
  gradientsW :/= tmp2
}
else {
  gradientsW := wFlat
  gradientsW *= outputSize * rawError.cols * lambda._1
  val tmp2 = wFlat :* wFlat
  tmp2 += epsilon
  sqrt.inPlace(tmp2)
  gradientsW :/= tmp2
  gradientsW +=  wFlat * (outputSize * rawError.cols * lambda._2)
}*/

// TODO: Add fast version that bypasses square roots.
/**
  *                           /--------
  * J(w_a) = c | w_a | = c   /    2
  *                        \/  w_a  + e
  *
  * epsilon will make the function smooth, which could be a desirable feature.
  *
  *        ---
  *        \
  * J(w) = /   J(w_i)
  *        ---
  *         i
  *                           (      2   )
  * d J(w_a)       1      c d ( w_a  + e )
  * -------- = -------- * ----------------
  *  d w_a     2 J(w_a)         d w_a
  *
  *                1
  *          = -------- * c 2 w_a
  *            2 J(w_a)
  *
  *            c w_a
  *          = ------
  *            J(w_a)
  *
  *   d J(w_a)
  * ----------- = 0
  * d w_b, a!=b
  *
  *                                          {              c  w_a          w_a
  *                                          { if e = 0 => --------- da = -------
  *            ---                           {             c | w_a |      | w_a |
  * D J(w_a)   \   d J(w_a)      c w_a       {
  * -------- = /   -------- di = ------ da = {                 c  w_a           w_a
  *  D w_a     ---   d_i         J(w_a)      { else     => -------------- = ------------
  *             i                            {                  /--------      /--------
  *                                          {             c   /    2         /    2
  *                                          {               \/  w_a  + e   \/  w_a  + e
  */
abstract class L1WeightDecay
  extends WeightDecay[L1WeightDecayBuilder] {

  val epsilon
  : Real = builder.epsilon

}

final class L1WeightDecayBuilder
  extends WeightDecayBuilder[L1WeightDecayBuilder] {

  override def repr
  : L1WeightDecayBuilder = this

  private var _epsilon
  : Real = 1.00000003e-5f

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(_epsilon >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : L1WeightDecayBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[L1WeightDecayBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: L1WeightDecayBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : L1WeightDecayBuilder = L1WeightDecayBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: L1WeightDecayBuilder =>
        other._epsilon = _epsilon
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def build(platformHint: Option[Platform],
                     seed:         InstanceSeed)
  : Regularizer = L1WeightDecayBuilder.lookupAndBuild(this, platformHint, seed)

}

object L1WeightDecayBuilder
  extends RegularizerVariantTable[L1WeightDecayBuilder] {

  register( 2, L1Regularizer_JVM_Baseline_Description)
  register(64, L1WeightDecay_Generic_Baseline_Description)

  final def apply()
  : L1WeightDecayBuilder = new L1WeightDecayBuilder

  final def apply(scaleCoefficient: ParameterBuilder)
  : L1WeightDecayBuilder = apply().setScaleCoefficient(scaleCoefficient)

  final def apply(scaleCoefficient: ParameterBuilder,
                  baseScope:        NullBuffer)
  : L1WeightDecayBuilder = apply(
    scaleCoefficient
  ).setBaseScope(baseScope)

  final def apply(scaleCoefficient: ParameterBuilder,
                  baseScope:        NullBuffer,
                  epsilon:          Real)
  : L1WeightDecayBuilder = apply(
    scaleCoefficient,
    baseScope
  ).setEpsilon(epsilon)



}