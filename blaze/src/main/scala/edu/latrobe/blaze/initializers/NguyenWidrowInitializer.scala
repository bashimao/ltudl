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

package edu.latrobe.blaze.initializers

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * This is a quite old idea for scaling the weights that tends to work quite
  * well for simple networks. Not sure why I picked that up. I stole this code
  * from the old C# NN I did a couple of years ago.
  *
  * Important quote from the web:
  * http://stackoverflow.com/questions/11868337/neural-network-initialization-nguyen-widrow-implementation
  *
  * "Nguyen & Widrow in their paper assume that the inputs are between -1 and
  * +1. Nguyen Widrow initialization is valid for any activation function which
  * is finite in length. Again in their paper they are only talking about a 2
  * layer NN, not sure about a 5 layer one.
  *                                           1
  *        beta * w                          ---
  *                i,j                        n
  * w   = -------------, where beta = 0.7 * m   , where n = no input nodes, m = no output nodes
  *  i,j    || w  ||
  *         ||  j ||
  *                 2
  *
  * Nice example code available at:
  * http://www.codeproject.com/Articles/38933/CNeuralNetwork-Make-Your-Neural-Network-Learn-Fast
  */
final class NguyenWidrowInitializer(override val builder: NguyenWidrowInitializerBuilder,
                                    override val seed:    InstanceSeed)
  extends BoostingInitializer[NguyenWidrowInitializerBuilder] {

  override def computeFanFactor(weights:       ValueTensor,
                                inputFanSize:  Int,
                                outputFanSize: Int)
  : Real = {
    val n = Math.pow(outputFanSize, 1.0 / inputFanSize)
    val d = weights.l2Norm(0.0)
    Real(n / d)
  }

}

final class NguyenWidrowInitializerBuilder
  extends BoostingInitializerBuilder[NguyenWidrowInitializerBuilder] {

  override def repr
  : NguyenWidrowInitializerBuilder = this

  override def defaultGain()
  : Real = 0.7f

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[NguyenWidrowInitializerBuilder]

  override protected def doCopy()
  : NguyenWidrowInitializerBuilder = NguyenWidrowInitializerBuilder()

  override def build(seed: InstanceSeed)
  : NguyenWidrowInitializer = new NguyenWidrowInitializer(this, seed)

}

object NguyenWidrowInitializerBuilder {

  final def apply()
  : NguyenWidrowInitializerBuilder = new NguyenWidrowInitializerBuilder()

  final def apply(gain: Real)
  : NguyenWidrowInitializerBuilder = apply().setGain(gain)

  final def apply(gain: Real, source: InitializerBuilder)
  : NguyenWidrowInitializerBuilder = apply(gain).setSource(source)

}
