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

import scala.language.experimental.macros
import scala.language.{higherKinds, implicitConversions}
import spire.macros._
import spire.macros.compat._

object ArrayMacros {

  final def foreachMacro1[T](c: Context)
                            (dst0: c.Expr[Array[T]], offset0: c.Expr[Int], stride0: c.Expr[Int],
                             length: c.Expr[Int])
                            (fn: c.Expr[T => Unit])
  : c.Expr[Unit] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val off0 = util.name("off0")
    val end0 = util.name("end0")

    val tree
    : Tree = {
      q"""
      if ($stride0 == 1) {
        if ($offset0 == 0) {
          var $off0 = 0
          while ($off0 < $length) {
            $fn($dst0($off0))
            $off0 += 1
          }
        }
        else {
          var $off0 = $offset0
          val $end0 = $offset0 + $length
          while ($off0 < $end0) {
            $fn($dst0($off0))
            $off0 += 1
          }
        }
      }
      else {
        var $off0 = $offset0
        val $end0 = $offset0 + $length * $stride0
        while ($off0 != $end0) {
          $fn($dst0($off0))
          $off0 += $stride0
        }
      }
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Unit](tree)
  }

  @inline
  final def foreach[T](dst0: Array[T], offset0: Int, stride0: Int,
                       length: Int)
                      (fn: T => Unit)
  : Unit = macro foreachMacro1[T]

  final def l2NormSqDoubleMacro(c: Context)
                               (array:   c.Expr[Array[Double]],
                                offset0: c.Expr[Int],
                                length:  c.Expr[Int])
  : c.Expr[Double] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val res = util.name("res")
    val off = util.name("off")
    val end = util.name("end")
    val tmp = util.name("tmp")

    val tree
    : Tree = {
      q"""
      var $res = 0.0
      var $off = $offset0
      val $end = $offset0 + $length
      while ($off < $end) {
        val $tmp = $array($off)
        $res += $tmp * $tmp
        $off += 1
      }
      $res
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Double](tree)
  }

  @inline
  final def l2NormSq(array: Array[Double], offset0: Int, length: Int)
  : Double = macro l2NormSqDoubleMacro

  final def l2NormSqFloatMacro(c: Context)
                              (array:   c.Expr[Array[Float]],
                               offset0: c.Expr[Int],
                               length:  c.Expr[Int])
  : c.Expr[Float] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val res = util.name("res")
    val off = util.name("off")
    val end = util.name("end")
    val tmp = util.name("tmp")

    val tree
    : Tree = {
      q"""
      var $res = 0.0f
      var $off = $offset0
      val $end = $offset0 + $length
      while ($off < $end) {
        val $tmp = $array($off)
        $res += $tmp * $tmp
        $off += 1
      }
      $res
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Float](tree)
  }

  @inline
  final def l2NormSq(array: Array[Float], offset0: Int, length: Int)
  : Float = macro l2NormSqFloatMacro

  final def transformMacro1[T](c: Context)
                              (dst0: c.Expr[Array[T]], offset0: c.Expr[Int], stride0: c.Expr[Int],
                               length: c.Expr[Int])
                              (fn: c.Expr[T => T])
  : c.Expr[Unit] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val off0 = util.name("off0")
    val end0 = util.name("end0")

    val tree
    : Tree = {
      q"""
      if ($stride0 == 1) {
        if ($offset0 == 0) {
          var $off0 = 0
          while ($off0 < $length) {
            $dst0($off0) = $fn($dst0($off0))
            $off0 += 1
          }
        }
        else {
          var $off0 = $offset0
          val $end0 = $offset0 + $length
          while ($off0 < $end0) {
            $dst0($off0) = $fn($dst0($off0))
            $off0 += 1
          }
        }
      }
      else {
        var $off0 = $offset0
        val $end0 = $offset0 + $length * $stride0
        while ($off0 != $end0) {
          $dst0($off0) = $fn($dst0($off0))
          $off0 += $stride0
        }
      }
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Unit](tree)
  }

  @inline
  final def transform[T](dst0: Array[T], offset0: Int, stride0: Int,
                         length: Int)
                        (fn: T => T)
  : Unit = macro transformMacro1[T]

  final def transformMacro2[T, U](c: Context)
                                 (dst0: c.Expr[Array[T]], offset0: c.Expr[Int], stride0: c.Expr[Int],
                                  src1: c.Expr[Array[U]], offset1: c.Expr[Int], stride1: c.Expr[Int],
                                  length: c.Expr[Int])
                                 (fn: c.Expr[(T, U) => T])
  : c.Expr[Unit] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val off1 = util.name("off1")
    val off0 = util.name("off0")
    val end0 = util.name("end0")

    val tree
    : Tree = {
      q"""
      if ($stride0 == 1 && $stride1 == 1) {
        if ($offset0 == $offset1) {
          if ($offset0 == 0) {
            var $off0 = 0
            while ($off0 < $length) {
              $dst0($off0) = $fn($dst0($off0), $src1($off1))
              $off0 += 1
            }
          }
          else {
            var $off0 = $offset0
            val $end0 = $offset0 + $length
            while ($off0 < $end0) {
              $dst0($off0) = $fn($dst0($off0), $src1($off1))
              $off0 += 1
            }
          }
        }
        else {
          var $off1 = $offset1
          var $off0 = $offset0
          val $end0 = $offset0 + $length
          while ($off0 < $end0) {
            $dst0($off0) = $fn($dst0($off0), $src1($off1))
            $off1 += 1
            $off0 += 1
          }
        }
      }
      else {
        var $off1 = $offset1
        var $off0 = $offset0
        val $end0 = $offset0 + $length * $stride0
        while ($off0 != $end0) {
          $dst0($off0) = $fn($dst0($off0), $src1($off1))
          $off1 += $stride1
          $off0 += $stride0
        }
      }
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Unit](tree)
  }

  @inline
  final def transform[T, U](dst0: Array[T], offset0: Int, stride0: Int,
                            src1: Array[U], offset1: Int, stride1: Int,
                            length: Int)
                           (fn: (T, U) => T)
  : Unit = macro transformMacro2[T, U]

}
