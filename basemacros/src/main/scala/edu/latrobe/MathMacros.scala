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

package edu.latrobe

import spire.macros._
import spire.macros.compat._

import scala.language.experimental.macros
import scala.language.{higherKinds, implicitConversions}

object MathMacros {

  // An approximate way to compute exp. Works well for small numbers. However,
  // Large numbers could become a problem.
  //
  // precision 30 = about -> +-0.00001 for [-100.0, 100.0]
  //
  // precision should be in [0, 64[
  final def approxExpMacroDouble(c: Context)
                                (x: c.Expr[Double], n: c.Expr[Int])
  : c.Expr[Double] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val twoPowerNInv = util.name("twoPowerNInv")
    val y            = util.name("y")
    val i            = util.name("i")

    val tree
    : Tree = {
      q"""
      val $twoPowerNInv: Double = 1.0 / (1L << $n)
      var $y = 1.0 + $x * $twoPowerNInv
      var $i = 0
      while ($i < $n) {
        $y *= $y
        $i += 1
      }
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def approxExp(x: Double, n: Int)
  : Double = macro approxExpMacroDouble

  final def approxExpMacroDouble4(c: Context)
                                 (x: c.Expr[Double])
  : c.Expr[Double] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val twoPower4Inv = util.name("twoPower4Inv")
    val y            = util.name("y")

    val tree
    : Tree = {
      q"""
      val $twoPower4Inv: Double = 1.0 / 16.0
      var $y = 1.0 + $x * $twoPower4Inv
      $y *= $y; $y *= $y; $y *= $y; $y *= $y // 0 ... 3
      $y
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def approxExp4(x: Double)
  : Double = macro approxExpMacroDouble4

  final def approxExpMacroDouble8(c: Context)
                                 (x: c.Expr[Double])
  : c.Expr[Double] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val twoPower8Inv = util.name("twoPower8Inv")
    val y            = util.name("y")

    val tree
    : Tree = {
      q"""
      val $twoPower8Inv: Double = 1.0 / 256.0
      var $y = 1.0 + $x * $twoPower8Inv
      $y *= $y; $y *= $y; $y *= $y; $y *= $y // 0 ... 3
      $y *= $y; $y *= $y; $y *= $y; $y *= $y // 4 ... 7
      $y
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def approxExp8(x: Double)
  : Double = macro approxExpMacroDouble8

  final def approxExpMacroDouble16(c: Context)
                                  (x: c.Expr[Double])
  : c.Expr[Double] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val twoPower16Inv = util.name("twoPower16Inv")
    val y             = util.name("y")

    val tree
    : Tree = {
      q"""
      val $twoPower16Inv: Double = 1.0 / 65536.0
      var $y = 1.0 + $x * $twoPower16Inv
      $y *= $y; $y *= $y; $y *= $y; $y *= $y // 0 ... 3
      $y *= $y; $y *= $y; $y *= $y; $y *= $y // 4 ... 7
      $y *= $y; $y *= $y; $y *= $y; $y *= $y // 8 ... B
      $y *= $y; $y *= $y; $y *= $y; $y *= $y // C ... F
      $y
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def approxExp16(x: Double)
  : Double = macro approxExpMacroDouble16

  final def approxSqrtMacroDouble(c: Context)
                                 (x: c.Expr[Double])
  : c.Expr[Double] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val tmp = util.name("tmp")
    val y = util.name("y")

    val tree
    : Tree = {
      q"""
      val $tmp = Double.doubleToLongBits($x) >> 32;
      val $y   = Double.longBitsToDouble(($tmp + 1072632448) << 31);
      // repeat the following line for more precision
      ($y + $x / $y) * 0.5;
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def approxSqrt(x: Double)
  : Double = macro approxSqrtMacroDouble

  final def lerpMacroDouble(c: Context)
                           (a: c.Expr[Double], b: c.Expr[Double], t: c.Expr[Double])
  : c.Expr[Double] = {
    import c.universe._

    val tree
    : Tree = q"$a + ($b - $a) * $t"

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def lerp(a: Double, b: Double, t: Double)
  : Double = macro lerpMacroDouble

  final def lerpMacroFloat(c: Context)
                          (a: c.Expr[Float], b: c.Expr[Float], t: c.Expr[Float])
  : c.Expr[Float] = {
    import c.universe._

    val tree
    : Tree = q"$a + ($b - $a) * $t"

    new FixedInlineUtil[c.type](c).inlineAndReset[Float](tree)
  }

  final def lerp(a: Float, b: Float, t: Float)
  : Float = macro lerpMacroFloat

  final def maxMacroDouble(c: Context)
                          (a: c.Expr[Double], b: c.Expr[Double])
  : c.Expr[Double] = {
    import c.universe._

    val tree
    : Tree = q"if ($a > $b) $a else $b"

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def max(a: Double, b: Double)
  : Double = macro maxMacroDouble

  final def maxMacroFloat(c: Context)
                         (a: c.Expr[Float], b: c.Expr[Float])
  : c.Expr[Float] = {
    import c.universe._

    val tree
    : Tree = q"if ($a > $b) $a else $b"

    new FixedInlineUtil[c.type](c).inlineAndReset[Float](tree)
  }

  final def max(a: Float, b: Float)
  : Float = macro maxMacroFloat

  final def minMacroDouble(c: Context)
                          (a: c.Expr[Double], b: c.Expr[Double])
  : c.Expr[Double] = {
    import c.universe._

    val tree
    : Tree = q"if ($a < $b) $a else $b"

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  final def min(a: Double, b: Double)
  : Double = macro minMacroDouble

  final def minMacroFloat(c: Context)
                         (a: c.Expr[Float], b: c.Expr[Float])
  : c.Expr[Float] = {
    import c.universe._

    val tree
    : Tree = q"if ($a < $b) $a else $b"

    new FixedInlineUtil[c.type](c).inlineAndReset[Float](tree)
  }

  final def min(a: Float, b: Float)
  : Float = macro minMacroFloat

  final def toUnsignedIntMacro(c: Context)
                              (x: c.Expr[Byte])
  : c.Expr[Int] = {
    import c.universe._

    val tree
    : Tree = q"if ($x >= 0) $x else $x + 256"

    new FixedInlineUtil[c.type](c).inlineAndReset[Int](tree)
  }

  final def toUnsigned(x: Byte)
  : Int = macro toUnsignedIntMacro

}
