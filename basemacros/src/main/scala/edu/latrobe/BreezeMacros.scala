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

/*
import breeze.linalg._
import scala.language.experimental.macros
import scala.language.{higherKinds, implicitConversions}
import spire.macros._
import spire.macros.compat._

object BreezeMacros {

  final def mergeMacro_DV_DV[T, U](c: Context)
                                  (v0: c.Expr[DenseVector[T]], v1: c.Expr[DenseVector[U]])
                                  (fn: c.Expr[(T, U) => T])
  : c.Expr[Unit] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val i       = util.name("i")
    val data1   = util.name("data1")
    val stride1 = util.name("stride1")
    val offset1 = util.name("offset1")
    val data0   = util.name("data0")
    val stride0 = util.name("stride0")
    val offset0 = util.name("offset0")
    val length0 = util.name("length0")
    val tmp     = util.name("tmp")
    val end0    = util.name("end0")

    /*
    val transformExImpl = c.universe.reify(
      (v0: DenseVector[T], v1: DenseVector[U], fn: (T, U) => T) => {
        require(v0.length == v1.length)

        val length = v0.length

        val data1   = v1.data
        val stride1 = v1.stride
        val data0   = v0.data
        val stride0 = v0.stride

        if (stride0 == 1 && stride1 == 1) {
          val offset1 = v1.offset
          val offset0 = v0.offset

          cforRange(0 until length)(i => {
            val tmp = offset0 + i
            data0(tmp) = fn(data0(tmp), data1(offset1 + i))
          })
        }
        else {
          // TODO: Java range check can be eliminated in some cases!
          var offset1: Int = v1.offset
          var offset0: Int = v0.offset
          val end0: Int    = v0.offset + v0.stride * v0.length
          while (offset0 < end0) {
            data0(offset0) = fn(data0(offset0), data1(offset1))
            offset1 += stride1
            offset0 += stride0
          }
        }
      }
    )
    */

    val tree: Tree = {
      /*
      q"""
      $transformExImpl($v0, $v1, $fn)
      """
      */
      q"""
      val $data1   = $v1.data
      val $stride1 = $v1.stride
      val $data0   = $v0.data
      val $stride0 = $v0.stride
      val $length0 = $v0.length
      require($length0 == $v1.length)

      if ($stride0 == 1 && $stride1 == 1) {
         val $offset1 = $v1.offset
         val $offset0 = $v0.offset
         var $i       = 0
         while ($i < $length0) {
          val $tmp = $offset0 + $i
          $data0($tmp) = $fn($data0($tmp), $data1($offset1 + $i))
          $i += 1
        }
      }
      else {
        // TODO: Java range check can be eliminated in some cases!
        var $offset1 = $v1.offset
        var $offset0 = $v0.offset
        val $end0    = $v0.offset + $v0.stride * $v0.length
        while ($offset0 < $end0) {
          $data0($offset0) = $fn($data0($offset0), $data1($offset1))
          $offset1 += $stride1
          $offset0 += $stride0
        }
      }
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Unit](tree)
  }

  final def merge_DV_DV[T, U](v0: DenseVector[T], v1: DenseVector[U])
                             (fn: (T, U) => T)
  : Unit = macro mergeMacro_DV_DV[T, U]

  final def mergeMacro_DV_DV_DV[T, U, V](c: Context)
                                        (v0: c.Expr[DenseVector[T]], v1: c.Expr[DenseVector[U]], v2: c.Expr[DenseVector[V]])
                                        (fn: c.Expr[(T, U, V) => T])
  : c.Expr[Unit] = {
    import c.universe._

    val util = SyntaxUtil[c.type](c)
    val i       = util.name("i")
    val length  = util.name("length")
    val data0   = util.name("data0")
    val data1   = util.name("data1")
    val data2   = util.name("data2")
    val stride0 = util.name("stride0")
    val stride1 = util.name("stride1")
    val stride2 = util.name("stride2")
    val offset0 = util.name("offset0")
    val offset1 = util.name("offset1")
    val offset2 = util.name("offset2")
    val tmp     = util.name("tmp")
    val end0    = util.name("end0")

    val tree: Tree = {
      q"""
      val $length = $v0.length
      require($length == $v1.length)
      require($length == $v2.length)

      val $data2   = $v2.data
      val $stride2 = $v2.stride
      val $data1   = $v1.data
      val $stride1 = $v1.stride
      val $data0   = $v0.data
      val $stride0 = $v0.stride

      if ($stride0 == 1 && $stride1 == 1 == $stride2 == 1) {
         val $offset2 = $v2.offset
         val $offset1 = $v1.offset
         val $offset0 = $v0.offset
         var $i       = 0
         while ($i < $length) {
          val $tmp = $offset0 + $i
          $data0($tmp) = $fn($data0($tmp), $data1($offset1 + $i), $data2($offset2 + $i))
          $i += 1
        }
      }
      else {
        // TODO: Java range check can be eliminated in some cases!
        var $offset2 = $v2.offset
        var $offset1 = $v1.offset
        var $offset0 = $v0.offset
        val $end0    = $v0.offset + $v0.stride * $v0.length
        while ($offset0 < $end0) {
          $data0($offset0) = $fn($data0($offset0), $data1($offset1), $data2($offset2))
          $offset2 += $stride2
          $offset1 += $stride1
          $offset0 += $stride0
        }
      }
      """
    }

    new FixedInlineUtil[c.type](c).inlineAndReset[Unit](tree)
  }

  final def merge_DV_DV_DV[T, U, V](v0: DenseVector[T], v1: DenseVector[U], v2: DenseVector[V])
                                   (fn: (T, U, V) => T)
  : Unit = macro mergeMacro_DV_DV_DV[T, U, V]

}
*/