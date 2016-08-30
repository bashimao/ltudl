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

import breeze.collection.mutable.SparseArray
import breeze.stats.distributions._
import breeze.storage.Zero
import breeze.util.ArrayUtil
import it.unimi.dsi.fastutil.io._
import java.io._
import scala.collection._
import scala.collection.parallel.ForkJoinTasks
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.reflect._
import scala.util.hashing._

/**
  * Extension of the array class.
  */
// TODO: Add back automatic switching to BLAS.
object ArrayEx {

  @inline
  final def abs(dst0: Array[Real])
  : Unit = abs(
    dst0, 0, 1,
    dst0.length
  )

  @inline
  final def abs(dst0: Array[Real], offset0: Int, stride0: Int,
                length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    length
  )(Math.abs)

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Real],
                src1: Real)
  : Unit = add(
    dst0, 0, 1,
    src1,
    dst0.length
  )

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Real], offset0: Int, stride0: Int,
                src1: Real,
                length: Int)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          dst0(i) += src1
          i       += 1
        }
      }
      else {
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) += src1
          off0       += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) += src1
        off0       += stride0
      }
    }
  }

  /**
    * dst0 = dst0 + src1
    */
  /**
    * Componentwise add.
    */
  @inline
  final def add(dst0: Array[Real],
                src1: Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    add(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Real], offset0: Int, stride0: Int,
                src1: Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            dst0(off0) += src1(off0)
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) += src1(off0)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) += src1(off1)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) += src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Real],
                src1: SparseArray[Real])
  : Unit = transformActive(
    dst0,
    src1
  )(_ + _)

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Real], offset0: Int, stride0: Int,
                src1: SparseArray[Real])
  : Unit = transformActive(
    dst0, offset0, stride0,
    src1
  )(_ + _)

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Int],
                src1: Int)
  : Unit = transform(
    dst0
  )(_ + src1)

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Int],
                src1: Array[Int])
  : Unit = transform(
    dst0,
    src1
  )(_ + _)

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Int], offset0: Int, stride0: Int,
                src1: Array[Int], offset1: Int, stride1: Int,
                length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ + _)

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Int],
                src1: SparseArray[Int])
  : Unit = transformActive(
    dst0,
    src1
  )(_ + _)

  /**
    * dst0 = dst0 + src1
    */
  @inline
  final def add(dst0: Array[Int], offset0: Int, stride0: Int,
                src1: SparseArray[Int])
  : Unit = transformActive(
    dst0, offset0, stride0,
    src1
  )(_ + _)

  /**
    * dst0 = alpha * dst0 + src1
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real],
                src1:  Real)
  : Unit = add(
    alpha,
    dst0, 0, 1,
    src1,
    dst0.length
  )

  /**
    * dst0 = alpha * dst0 + src1
    */
  @inline
  final def add(alpha:  Real,
                dst0:   Array[Real], offset0: Int, stride0: Int,
                src1:   Real,
                length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    length
  )(alpha * _ + src1)

  /**
    * dst0 = alpha * dst0 + src1
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real],
                src1:  Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    add(
      alpha,
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = alpha * dst0 + src1
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real], offset0: Int, stride0: Int,
                src1:  Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = alpha * dst0(i) + src1(i)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = alpha * dst0(off0) + src1(off0)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = alpha * dst0(off0) + src1(off1)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = alpha * dst0(off0) + src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = dst0 + beta * src1
    */
  @inline
  final def add(dst0: Array[Real],
                beta: Real,
                src1: Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    add(
      dst0, 0, 1,
      beta,
      src1, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = dst0 + beta * src1
    */
  @inline
  final def add(dst0: Array[Real], offset0: Int, stride0: Int,
                beta: Real,
                src1: Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) += beta * src1(i)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) += beta * src1(off0)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) += beta * src1(off1)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) += beta * src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = alpha * dst0 + beta * src1
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real],
                beta:  Real,
                src1:  Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    add(
      alpha,
      dst0, 0, 1,
      beta,
      src1, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = alpha * dst0 + beta * src1
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real], offset0: Int, stride0: Int,
                beta:  Real,
                src1:  Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            dst0(off0) = alpha * dst0(off0) + beta * src1(off0)
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = alpha * dst0(off0) + beta * src1(off0)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = alpha * dst0(off0) + beta * src1(off1)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = alpha * dst0(off0) + beta * src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = dst0 + src1 * src2
    */
  @inline
  final def add(dst0: Array[Real],
                src1: Array[Real],
                src2: Array[Real])
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    add(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = dst0 + src1 * src2
    */
  @inline
  final def add(dst0: Array[Real], offset0: Int, stride0: Int,
                src1: Array[Real], offset1: Int, stride1: Int,
                src2: Array[Real], offset2: Int, stride2: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == offset1 && offset0 == offset2) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            dst0(off0) += src1(off0) * src2(off0)
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) += src1(off0) * src2(off0)
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) += src1(off1) * src2(off2)
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) += src1(off1) * src2(off2)
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = alpha * dst0 + src1 * src2
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real],
                src1:  Array[Real],
                src2:  Array[Real],
                length: Int)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    add(
      alpha,
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = alpha * dst0 + src1 * src2
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real], offset0: Int, stride0: Int,
                src1:  Array[Real], offset1: Int, stride1: Int,
                src2:  Array[Real], offset2: Int, stride2: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == offset1 && offset0 == offset2) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            dst0(off0) = alpha * dst0(off0) + src1(off0) * src2(off0)
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = alpha * dst0(off0) + src1(off0) * src2(off0)
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = alpha * dst0(off0) + src1(off1) * src2(off2)
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = alpha * dst0(off0) + src1(off1) * src2(off2)
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = dst0 + beta * src1 * src2
    */
  @inline
  final def add(dst0: Array[Real],
                beta: Real,
                src1: Array[Real],
                src2: Array[Real],
                length: Int)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    add(
      dst0, 0, 1,
      beta,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = dst0 + beta * src1 * src2
    */
  @inline
  final def add(dst0: Array[Real], offset0: Int, stride0: Int,
                beta: Real,
                src1: Array[Real], offset1: Int, stride1: Int,
                src2: Array[Real], offset2: Int, stride2: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == offset1 && offset0 == offset2) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            dst0(off0) += beta * src1(off0) * src2(off0)
            off0       += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) += beta * src1(off0) * src2(off0)
            off0       += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) += beta * src1(off1) * src2(off2)
          off2       += 1
          off1       += 1
          off0       += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) += beta * src1(off1) * src2(off2)
        off2       += stride2
        off1       += stride1
        off0       += stride0
      }
    }
  }

  /**
    * dst0 = alpha * dst0 + beta * src1 * src2
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real],
                beta:  Real,
                src1:  Array[Real],
                src2:  Array[Real],
                length: Int)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    add(
      alpha,
      dst0, 0, 1,
      beta,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = alpha * dst0 + beta * src1 * src2
    */
  @inline
  final def add(alpha: Real,
                dst0:  Array[Real], offset0: Int, stride0: Int,
                beta:  Real,
                src1:  Array[Real], offset1: Int, stride1: Int,
                src2:  Array[Real], offset2: Int, stride2: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == offset1 && offset0 == offset2) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            dst0(off0) = alpha * dst0(off0) + beta * src1(off0) * src2(off0)
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = alpha * dst0(off0) + beta * src1(off0) * src2(off0)
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = alpha * dst0(off0) + beta * src1(off1) * src2(off2)
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = alpha * dst0(off0) + beta * src1(off1) * src2(off2)
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def compare[T, U](src0: Array[T],
                          src1: Array[U])
  : Boolean = !exists(
    src0,
    src1
  )(_ != _)

  @inline
  final def compare[T, U](src0: Array[T], offset0: Int, stride0: Int,
                          src1: Array[U], offset1: Int, stride1: Int,
                          length: Int)
  : Boolean = !exists(
    src0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ != _)

  @inline
  final def concat[T](src0: Array[T],
                      src1: Array[T])
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.length + src1.length)
    concatEx(
      result, 0, 1,
      src0, 0, 1,
      src0.length,
      src1, 0, 1,
      src1.length
    )
    result
  }

  /**
    * Normal vertical concatenation.
    */
  @inline
  final def concat[T](src0: SparseArray[T],
                      src1: SparseArray[T])
                     (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : SparseArray[T] = {
    // Create shorthands for frequently used variables.
    val length1  = src1.size
    val indices1 = src1.index
    val data1    = src1.data
    val used1    = src1.activeSize
    val length0  = src0.size
    val indices0 = src0.index
    val data0    = src0.data
    val used0    = src0.activeSize

    // Allocate and fill index buffers.
    val usedR    = used0 + used1
    val indicesR = new Array[Int](usedR)
    set(
      indicesR, 0, 1,
      indices0, 0, 1,
      used0
    )
    var i = 0
    while (i < used1) {
      indicesR(used0 + i) = length0 + indices1(i)
      i += 1
    }

    // Allocate and fill data buffers.
    val dataR = new Array[T](usedR)
    set(
      dataR, 0, 1,
      data0, 0, 1,
      used0
    )
    set(
      dataR, used0, 1,
      data1, 0,     1,
      used1
    )
    new SparseArray(indicesR, dataR, usedR, length0 + length1, zeroT.zero)
  }

  @inline
  final def concat[T](src0: Array[T],
                      src1: Array[Array[T]])
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.length + foldLeft(0, src1)(_ + _.length))
    val offset = concatEx(
      result, 0, 1,
      src0,
      src1
    )
    assume(offset == result.length)
    result
  }

  @inline
  final def concat[T](src0: Array[T],
                      src1: Traversable[Array[T]])
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.length + src1.foldLeft(0)(_ + _.length))
    val offset = concatEx(
      result, 0, 1,
      src0,
      src1
    )
    assume(offset == result.length)
    result
  }

  @inline
  final def concat[T](src0: Array[Array[T]])
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](foldLeft(0, src0)(_ + _.length))
    val offset = concatEx(
      result, 0, 1,
      src0
    )
    assume(offset == result.length)
    result
  }

  @inline
  final def concat[T](src0: Traversable[Array[T]])
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.foldLeft(0)(_ + _.length))
    val offset = concatEx(
      result, 0, 1,
      src0
    )
    assume(offset == result.length)
    result
  }

  @inline
  final def concatEx[T](dst0: Array[T],
                        src1: Array[T],
                        src2: Array[T])
  : Unit = {
    require(dst0.length == src1.length + src2.length)
    concatEx(
      dst0, 0, 1,
      src1, 0, 1,
      src1.length,
      src2, 0, 1,
      src2.length
    )
  }

  @inline
  final def concatEx[T](dst0:    Array[T], offset0: Int, stride0: Int,
                        src1:    Array[T], offset1: Int, stride1: Int,
                        length1: Int,
                        src2:    Array[T], offset2: Int, stride2: Int,
                        length2: Int)
  : Unit = {
    set(
      dst0, offset0, stride0,
      src1, offset1, stride1,
      length1
    )
    set(
      dst0, offset0 + stride0 * length1, stride0,
      src2, offset2,                     stride2,
      length2
    )
  }

  @inline
  final def concatEx[T](dst0:    Array[T], offset0: Int, stride0: Int,
                        src1:    Array[T], offset1: Int, stride1: Int,
                        length1: Int,
                        src2:    SparseArray[T])
  : Unit = {
    set(
      dst0, offset0, stride0,
      src1, offset1, stride1,
      length1
    )
    setActive(
      dst0, offset0 + stride0 * length1, stride0,
      src2
    )
  }

  @inline
  final def concatEx[T](dst0:    Array[T], offset0: Int, stride0: Int,
                        src1:    SparseArray[T],
                        src2:    Array[T], offset2: Int, stride2: Int,
                        length2: Int)
  : Unit = {
    setActive(
      dst0, offset0, stride0,
      src1
    )
    set(
      dst0, offset0 + stride0 * src1.length, stride0,
      src2, offset2,                         stride2,
      length2
    )
  }

  @inline
  final def concatEx[T](dst0: Array[T],
                        src1: Array[Array[T]])
  : Unit = {
    val offset = concatEx(
      dst0, 0, 1,
      src1
    )
    assume(offset == dst0.length)
  }

  @inline
  final def concatEx[T](dst0: Array[T], offset0: Int, stride0: Int,
                        src1: Array[Array[T]])

  : Int = {
    var off0 = offset0
    foreach(src1)(src1 => {
      set(
        dst0, off0, stride0,
        src1, 0,    1,
        src1.length
      )
      off0 += stride0 * src1.length
    })
    off0
  }

  @inline
  final def concatEx[T](dst0: Array[T],
                        src1: Traversable[Array[T]])
  : Unit = {
    val offset = concatEx(
      dst0, 0, 1,
      src1
    )
    assume(offset == dst0.length)
  }

  @inline
  final def concatEx[T](dst0: Array[T], offset0: Int, stride0: Int,
                        src1: Traversable[Array[T]])
  : Int = {
    var off0 = offset0
    src1.foreach(src1 => {
      set(
        dst0, off0, stride0,
        src1, 0,    1,
        src1.length
      )
      off0 += stride0 * src1.length
    })
    off0
  }

  @inline
  final def concatEx[T](dst0: Array[T],
                        src1: Array[T],
                        src2: Array[Array[T]])
  : Unit = {
    val offset = concatEx(
      dst0, 0, 1,
      src1,
      src2
    )
    assume(offset == dst0.length)
  }

  @inline
  final def concatEx[T](dst0: Array[T], offset0: Int, stride0: Int,
                        src1: Array[T],
                        src2: Array[Array[T]])
  : Int = {
    var off0 = offset0
    set(
      dst0, off0, stride0,
      src1, 0,    1,
      src1.length
    )
    off0 += stride0 * src1.length

    foreach(src2)(src2 => {
      set(
        dst0, off0, stride0,
        src2, 0,    1,
        src2.length
      )
      off0 += stride0 * src2.length
    })
    off0
  }

  @inline
  final def concatEx[T](dst0: Array[T],
                        src1: Array[T],
                        src2: TraversableOnce[Array[T]])
  : Unit = {
    val offset = concatEx(
      dst0, 0, 1,
      src1,
      src2
    )
    assume(offset == dst0.length)
  }

  @inline
  final def concatEx[T](dst0: Array[T], offset0: Int, stride0: Int,
                        src1: Array[T],
                        src2: TraversableOnce[Array[T]])
  : Int = {
    var off0 = offset0
    set(
      dst0, off0, stride0,
      src1, 0,    1,
      src1.length
    )
    off0 += stride0 * src1.length

    src2.foreach(src2 => {
      set(
        dst0, off0, stride0,
        src2, 0,    1,
        src2.length
      )
      off0 += stride0 * src2.length
    })
    off0
  }

  @inline
  final def contains[T](src0: Array[T],
                        src1: T)
  : Boolean = contains(
    src0, 0, 1,
    src1,
    src0.length
  )

  @inline
  final def contains[T](src0:   Array[T], offset0: Int, stride0: Int,
                        src1:   T,
                        length: Int)
  : Boolean = exists(
    src0, offset0, stride0,
    length
  )(_ == src1)

  @inline
  final def copy[T](src0: Array[T])
  : Array[T] = src0.clone()

  @inline
  final def copy[T](src0: Array[T], offset0: Int, stride0: Int,
                    length: Int)
                   (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](length)
    set(
      result, 0,       1,
      src0,   offset0, stride0,
      length
    )
    result
  }

  @inline
  final def count[T](src0: Array[T])
                    (predicate: T => Boolean)
  : Int = count(
    src0, 0, 1,
    src0.length
  )(predicate)

  @inline
  final def count[T](src0: Array[T], offset0: Int, stride0: Int,
                     length: Int)
                    (predicate: T => Boolean)
  : Int = {
    var result = 0
    foreach(
      src0, offset0, stride0,
      length
    )(value0 => {
      if (predicate(value0)) {
        result += 1
      }
    })
    result
  }

  @inline
  final def count[T](src0: SparseArray[T])
                    (predicate: T => Boolean)
  : Int = {
    var result = 0
    if (predicate(src0.default)) {
      result += src0.size - src0.activeSize
    }
    foreachActive(src0)(v0 => {
      if (predicate(v0)) {
        result += 1
      }
    })
    result
  }

  @inline
  final def countApprox[T](src0:      SparseArray[T],
                           rng:       PseudoRNG,
                           noSamples: Int)
                          (predicate: T => Boolean)
  : Int = {
    var result = 0
    if (predicate(src0.default)) {
      result += src0.size - src0.activeSize
    }
    result += countActiveApprox(
      src0, rng, noSamples
    )(predicate)
    result
  }

  @inline
  final def countApprox[T](src0:      Array[T],
                           rng:       PseudoRNG,
                           noSamples: Int)
                          (predicate: T => Boolean)
  : Int = countApprox(
    src0, 0, 1,
    src0.length,
    rng,
    noSamples
  )(predicate)

  @inline
  final def countApprox[T](src0:      Array[T], offset0: Int, stride0: Int,
                           length:    Int,
                           rng:       PseudoRNG,
                           noSamples: Int)
                          (predicate: T => Boolean)
  : Int = {
    if (length == 0) {
      0
    }
    else if (length <= noSamples) {
      count(
        src0, offset0, stride0,
        length
      )(predicate)
    }
    else {
      var count = 0L
      var i     = 0

      if (stride0 == 1) {
        while (i < noSamples) {
          val index = rng.nextInt(length)
          if (predicate(src0(index))) {
            count += 1L
          }
          i += 1
        }
      }
      else {
        while (i < noSamples) {
          val index = rng.nextInt(length) * stride0
          if (predicate(src0(index))) {
            count += 1L
          }
          i += 1
        }
      }

      ((count * src0.length + noSamples / 2) / noSamples).toInt
    }
  }

  @inline
  final def countActive[T](src0: SparseArray[T])
                          (predicate: T => Boolean)
  : Int = {
    var result = 0
    foreachActive(src0)(value0 => {
      if (predicate(value0)) {
        result += 1
      }
    })
    result
  }

  @inline
  final def countActiveApprox[T](src0:      SparseArray[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Int = countApprox(
    src0.data, 0, 1,
    src0.activeSize,
    rng,
    noSamples
  )(predicate)

  @inline
  final def createOutputStream()
  : FastByteArrayOutputStream = new FastByteArrayOutputStream()

  @inline
  final def deepCopy[T <: CopyableEx[T]](src0: Array[T])
                                        (implicit tagT: ClassTag[T])
  : Array[T] = deepCopy(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def deepCopy[T <: CopyableEx[T]](src0: Array[T], offset0: Int, stride0: Int,
                                         length: Int)
                                        (implicit tagT: ClassTag[T])
  : Array[T] = map(
    src0, offset0, stride0,
    length
  )(_.copy)

  @inline
  final def deepCopy[T <: CopyableEx[T] with ClosableEx](src0: Array[Ref[T]])
                                                        (implicit tagT: ClassTag[T])
  : Array[Ref[T]] = deepCopy(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def deepCopy[T <: CopyableEx[T] with ClosableEx](src0: Array[Ref[T]], offset0: Int, stride0: Int,
                                                         length: Int)
                                                        (implicit tagT: ClassTag[T])
  : Array[Ref[T]] = map(
    src0, offset0, stride0,
    length
  )(ref => Ref(ref.value.copy))

  @inline
  final def deserialize[T <: JSerializable](src0: Array[Byte])
  : T = deserialize[T](
    src0, 0, 1,
    src0.length
  )

  @inline
  final def deserialize[T <: JSerializable](src0: Array[Byte], offset0: Int, stride0: Int,
                                            length: Int)
  : T = {
    using(
      toObjectInputStream(
        src0, offset0, stride0,
        length
      )
    )(_.readObject().asInstanceOf[T])
  }

  @inline
  final def diffL2NormSq(src0: Array[Real], offset0: Int, stride0: Int,
                         value1: Real,
                         length: Int)
  : Real = foldLeft(
    Real.zero,
    src0, offset0, stride0,
    length
  )((res, value0) => {
    val tmp = value0 - value1
    res + tmp * tmp
  })

  @inline
  final def diffL2NormSq(src0: SparseArray[Real],
                         src1: Real)
  : Real = foldLeft(
    Real.zero,
    src0
  )((res, src0) => {
    val tmp = src0 - src1
    res + tmp * tmp
  })

  @inline
  final def divide(src0: Real,
                   dst1: Array[Real])
  : Unit = divide(
    src0,
    dst1, 0, 1,
    dst1.length
  )

  @inline
  final def divide(src0: Real,
                   dst1: Array[Real], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = transform(
    dst1, offset1, stride1,
    length
  )(src0 / _)

  @inline
  final def divide(dst0: Array[Real],
                   src1: Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    divide(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def divide(dst0: Array[Real], offset0: Int, stride0: Int,
                   src1: Array[Real], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ / _)

  @inline
  final def divide(epsilon0: Real,
                   dst0:     Array[Real],
                   src1:     Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    divide(
      epsilon0,
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def divide(epsilon0: Real,
                   dst0:     Array[Real], offset0: Int, stride0: Int,
                   src1:     Array[Real], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )((dst0, src1) => (dst0 + epsilon0) / src1)

  @inline
  final def divide(dst0:     Array[Real],
                   epsilon1: Real,
                   src1:     Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    divide(
      dst0, 0, 1,
      epsilon1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def divide(dst0:     Array[Real], offset0: Int, stride0: Int,
                   epsilon1: Real,
                   src1:     Array[Real], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )((dst0, src1) => dst0 / (src1 + epsilon1))

  @inline
  final def divide(epsilon0: Real,
                   dst0:     Array[Real],
                   epsilon1: Real,
                   src1:     Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    divide(
      epsilon0,
      dst0, 0, 1,
      epsilon1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def divide(epsilon0: Real,
                   dst0:     Array[Real], offset0: Int, stride0: Int,
                   epsilon1: Real,
                   src1:     Array[Real], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )((dst0, src1) => (dst0 + epsilon0) / (src1 + epsilon1))

  @inline
  final def divide(dst0: Array[Int],
                   src1: Array[Int])
  : Unit = {
    require(dst0.length == src1.length)
    divide(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def divide(dst0: Array[Int], offset0: Int, stride0: Int,
                   src1: Array[Int], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ / _)

  @inline
  final def dot(src0: Array[Real],
                src1: Array[Real])
  : Real = {
    require(src0.length == src1.length)
    dot(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )
  }

  @inline
  final def dot(src0:   Array[Real], offset0: Int, stride0: Int,
                src1:   Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Real = foldLeft(
    Real.zero,
    src0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ + _ * _)

  @inline
  final def exists[T](src0: Array[T])
                     (predicate: T => Boolean)
  : Boolean = exists(
    src0, 0, 1,
    src0.length
  )(predicate)

  @inline
  final def exists[T](src0:   Array[T], offset0: Int, stride0: Int,
                      length: Int)
                     (predicateFn: T => Boolean)
  : Boolean = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var off0 = 0
        while (off0 < length) {
          if (predicateFn(src0(off0))) {
            return true
          }
          off0 += 1
        }
      }
      else {
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          if (predicateFn(src0(off0))) {
            return true
          }
          off0 += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        if (predicateFn(src0(off0))) {
          return true
        }
        off0 += stride0
      }
    }
    false
  }

  @inline
  final def exists[T, U](src0: Array[T],
                         src1: Array[U])
                        (predicateFn: (T, U) => Boolean)
  : Boolean = {
    require(src0.length == src1.length)
    exists(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )(predicateFn)
  }

  @inline
  final def exists[T, U](src0: Array[T], offset0: Int, stride0: Int,
                         src1: Array[U], offset1: Int, stride1: Int,
                         length: Int)
                        (predicateFn: (T, U) => Boolean)
  : Boolean = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            if (predicateFn(src0(off0), src1(off0))) {
              return true
            }
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            if (predicateFn(src0(off0), src1(off0))) {
              return true
            }
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          if (predicateFn(src0(off0), src1(off1))) {
            return true
          }
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        if (predicateFn(src0(off0), src1(off1))) {
          return true
        }
        off1 += stride1
        off0 += stride0
      }
    }
    false
  }

  @inline
  final def fill[T](length: Int)
                   (fn: => T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](length)
    fill(
      result
    )(fn)
    result
  }

  @inline
  final def fill[T](length: Int, value: T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](length)
    ArrayUtil.fill(result, 0, result.length, value)
    result
  }

  @inline
  final def fill[T](length: Int, distribution: Distribution[T])
                   (implicit tagT: ClassTag[T])
  : Array[T] = fill(
    length
  )(distribution.sample())

  @inline
  final def fill[T](dst0: Array[T])
                   (fn: => T)
  : Unit = fill(
    dst0, 0, 1,
    dst0.length
  )(fn)

  @inline
  final def fill[T](dst0: Array[T], offset0: Int, stride0: Int,
                    length: Int)
                   (fn: => T)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var off0 = 0
        while (off0 < length) {
          dst0(off0) = fn
          off0 += 1
        }
      }
      else {
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = fn
          off0 += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn
        off0 += stride0
      }
    }
  }

  @inline
  final def fill[T](dst0: Array[T],
                    src1: T)
  : Unit = ArrayUtil.fill(
    dst0, 0,
    dst0.length,
    src1
  )

  @inline
  final def fill[T](dst0: Array[T], offset0: Int, stride0: Int,
                    src1: T,
                    length: Int)
  : Unit = {
    if (stride0 == 1) {
      ArrayUtil.fill(
        dst0, offset0,
        length,
        src1
      )
    }
    else {
      var off = offset0
      val end = offset0 + length * stride0
      while (off != end) {
        dst0(off) = src1
        off += stride0
      }
    }
  }

  @inline
  final def fill[T](dst0: Array[T],
                    src1: Distribution[T])
  : Unit = fill(
    dst0, 0, 1,
    src1,
    dst0.length
  )

  @inline
  final def fill[T](dst0: Array[T], offset0: Int, stride0: Int,
                    src1: Distribution[T],
                    length: Int)
  : Unit = fill(
    dst0, offset0, stride0,
    length
  )(src1.sample())

  @inline
  final def fill[T, U](src0: Array[T],
                       dst1: Array[U])
                      (fn: U => T)
  : Unit = {
    require(src0.length == dst1.length)
    fill(
      src0, 0, 1,
      dst1, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def fill[T, U](dst0: Array[T], offset0: Int, stride0: Int,
                       src1: Array[U], offset1: Int, stride1: Int,
                       length: Int)
                      (fn: U => T)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            dst0(off0) = fn(src1(off0))
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = fn(src1(off0))
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = fn(src1(off1))
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn(src1(off1))
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def filter[T](src0: Array[T])
                     (predicateFn: T => Boolean)
                     (implicit tagT: ClassTag[T])
  : Array[T] = filter(
    src0, 0, 1,
    src0.length
  )(predicateFn)

  @inline
  final def filter[T](src0: Array[T], offset0: Int, stride0: Int,
                      length: Int)
                     (predicateFn: T => Boolean)
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val builder = Array.newBuilder[T]
    foreach(
      src0, offset0, stride0,
      length
    )(value0 => {
      if (predicateFn(value0)) {
        builder += value0
      }
    })
    builder.result()
  }

  /**
    * Keeps only values where conditionFn evaluates true.
    */
  @inline
  final def filter[T](dst0: SparseArray[T])
                     (predicateFn: T => Boolean)
  : Unit = {
    val indices0 = dst0.index
    val data0    = dst0.data
    val used0    = dst0.activeSize
    var newUsed0 = 0
    var i        = 0
    while (i < used0) {
      val value = data0(i)
      if (predicateFn(value)) {
        data0(newUsed0)    = value
        indices0(newUsed0) = indices0(i)
        newUsed0 += 1
      }
      i += 1
    }
    dst0.use(indices0, data0, newUsed0)
  }

  @inline
  final def find[T](src0: Array[T])
                   (predicate: T => Boolean)
  : Option[T] = find(
    src0, 0, 1,
    src0.length
  )(predicate)

  @inline
  final def find[T](src0: Array[T], offset0: Int, stride0: Int,
                    length: Int)
                   (predicateFn: T => Boolean)
  : Option[T] = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var off0 = 0
        while (off0 < length) {
          val tmp = src0(off0)
          if (predicateFn(tmp)) {
            return Some(tmp)
          }
          off0 += 1
        }
      }
      else {
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          val tmp = src0(off0)
          if (predicateFn(tmp)) {
            return Some(tmp)
          }
          off0 += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        val tmp = src0(off0)
        if (predicateFn(tmp)) {
          return Some(tmp)
        }
        off0 += stride0
      }
    }
    None
  }

  @inline
  final def finishAll[T](src0: Array[Future[T]])
  : Unit = finishAll(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def finishAll[T](src0: Array[Future[T]], offset0: Int, stride0: Int,
                         length: Int)
  : Unit = ArrayEx.foreach(
    src0, offset0, stride0,
    length
  )(FutureEx.finish)

  @inline
  final def flatten(src0: Array[String])
  : String = {
    val builder = StringBuilder.newBuilder
    foreach(
      src0
    )(builder ++= _)
    builder.result()
  }

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: Array[U])
                          (fn: (T, U) => T)
  : T = {
    var result = src0
    foreach(
      src1
    )(v1 => result = fn(result, v1))
    result
  }

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: Array[U], offset1: Int, stride1: Int,
                           length: Int)
                          (fn: (T, U) => T)
  : T = {
    var result = src0
    foreach(
      src1, offset1, stride1,
      length
    )(v1 => result = fn(result, v1))
    result
  }

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: SparseArray[U])
                          (fn: (T, U) => T)
  : T = {
    var result = src0
    foreach(
      src1
    )(v1 => result = fn(result, v1))
    result
  }

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: Array[U],
                              src2: Array[V])
                             (fn: (T, U, V) => T)
  : T = {
    var result = src0
    foreach(
      src1,
      src2
    )((v1, v2) => result = fn(result, v1, v2))
    result
  }

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: Array[U], offset1: Int, stride1: Int,
                              src2: Array[V], offset2: Int, stride2: Int,
                              length: Int)
                             (fn: (T, U, V) => T)
  : T = {
    var result = src0
    foreach(
      src1, offset1, stride1,
      src2, offset2, stride2,
      length
    )((v1, v2) => result = fn(result, v1, v2))
    result
  }

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: Array[U], offset1: Int, stride1: Int,
                              src2: SparseArray[V])
                             (fn: (T, U, V) => T)
  : T = {
    var result = src0
    foreach(
      src1, offset1, stride1,
      src2
    )((v1, v2) => result = fn(result, v1, v2))
    result
  }

  @inline
  final def foldLeft[T, U, V, W](src0: T,
                                 src1: Array[U], offset1: Int, stride1: Int,
                                 src2: Array[V], offset2: Int, stride2: Int,
                                 src3: Array[W], offset3: Int, stride3: Int,
                                 length: Int)
                                (fn: (T, U, V, W) => T)
  : T = {
    var result = src0
    foreach(
      src1, offset1, stride1,
      src2, offset2, stride2,
      src3, offset3, stride3,
      length
    )((v1, v2, v3) => result = fn(result, v1, v2, v3))
    result
  }

  @inline
  final def foldLeft[T, U](src0: Array[T])
                          (fnHead: T => U, fnTail: (U, T) => U)
  : U = {
    var result = fnHead(src0(0))
    var i      = 1
    while (i < src0.length) {
      result = fnTail(result, src0(i))
      i += 1
    }
    result
  }

  @inline
  final def foldLeftActive[T, U](src0: T,
                                 src1: SparseArray[U])
                                (fn: (T, U) => T)
  : T = {
    var result = src0
    foreachActive(
      src1
    )(v1 => result = fn(result, v1))
    result
  }

  @inline
  final def foldLeftEx[T, U](src0: T,
                             src1: SparseArray[U])
                            (fn0: (T, U) => T, fn1: T => T)
  : T = {
    var result = src0
    foreachEx(
      src1
    )(
      v1 => result = fn0(result, v1),
      () => result = fn1(result)
    )
    result
  }

  @inline
  final def foldLeftEx[T, U, V](src0: T,
                                src1: Array[U], offset1: Int, stride1: Int,
                                src2: SparseArray[V])
                               (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : T = {
    var result = src0
    foreachEx(
      src1, offset1, stride1,
      src2
    )(
      (v1, v2) => result = fn0(result, v1, v2),
      (v1)     => result = fn1(result, v1)
    )
    result
  }

  @inline
  final def foldLeftPairs[T, U](src0: T,
                                src1: Array[U])
                               (fn: (T, Int, U) => T)
  : T = {
    var result = src0
    foreachPair(
      src1
    )((i, v0) => result = fn(result, i, v0))
    result
  }

  @inline
  final def foldLeftPairs[T, U, V](src0: T,
                                   src1: Array[U],
                                   src2: Array[V])
                                  (fn: (T, Int, U, V) => T)
  : T = {
    var result = src0
    foreachPair(
      src1,
      src2
    )((i, v1, v2) => result = fn(result, i, v1, v2))
    result
  }

  @inline
  final def foreach[T](src0: Array[T])
                      (fn: T => Unit)
  : Unit = foreach(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def foreach[T](src0: SparseArray[T])
                      (fn: T => Unit)
  : Unit = {
    foreachEx(
      src0
    )(fn, fn(src0.default))
  }

  @inline
  final def foreach[T](src0: Array[T], offset0: Int, stride0: Int,
                       length: Int)
                      (fn: T => Unit)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var off0 = 0
        while (off0 < length) {
          fn(src0(off0))
          off0 += 1
        }
      }
      else {
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          fn(src0(off0))
          off0 += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        fn(src0(off0))
        off0 += stride0
      }
    }
  }

  @inline
  final def foreach[T, U](src0: Array[T],
                          src1: Array[U])
                         (fn: (T, U) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    foreach(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreach[T, U](src0: Array[T], offset0: Int, stride0: Int,
                          src1: Array[U], offset1: Int, stride1: Int,
                          length: Int)
                         (fn: (T, U) => Unit)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            fn(src0(off0), src1(off0))
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            fn(src0(off0), src1(off0))
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          fn(src0(off0), src1(off1))
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        fn(src0(off0), src1(off1))
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def foreach[T, U](src0: Array[T], offset0: Int, stride0: Int,
                          src1: SparseArray[U])
                         (fn: (T, U) => Unit)
  : Unit = foreachEx(
    src0, offset0, stride0,
    src1
  )(fn, fn(_, src1.default))

  @inline
  final def foreach[T, U, V](src0: Array[T],
                             src1: Array[U],
                             src2: Array[V])
                            (fn: (T, U, V) => Unit)
  : Unit = {
    require(
      src0.length == src1.length &&
      src1.length == src2.length
    )
    foreach(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreach[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                             src1: Array[U], offset1: Int, stride1: Int,
                             src2: Array[V], offset2: Int, stride2: Int,
                             length: Int)
                            (fn: (T, U, V) => Unit)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (
        offset0 == offset1 &&
        offset0 == offset2
      ) {
        if (offset0 == 0) {
          var off0 = 0
          while (off0 < length) {
            fn(src0(off0), src1(off0), src2(off0))
            off0 += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            fn(src0(off0), src1(off0), src2(off0))
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          fn(src0(off0), src1(off1), src2(off2))
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length + stride0
      while (off0 != end0) {
        fn(src0(off0), src1(off1), src2(off2))
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def foreach[T](src0: Array[T],
                       indices: PseudoRNG, noSamplesMax: Int)
                      (fn: T => Unit)
  : Unit = foreach(
    src0, 0, 1,
    src0.length,
    indices, noSamplesMax
  )(fn)

  @inline
  final def foreach[T](src0:   Array[T], offset0: Int, stride0: Int,
                       length: Int,
                       indices: PseudoRNG, noSamplesMax: Int)
                      (fn: T => Unit)
  : Unit = {
    if (noSamplesMax < length) {
      if (offset0 == 0) {
        if (stride0 == 1) {
          var i = 0
          while (i < noSamplesMax) {
            val off0 = indices.nextInt(length)
            fn(src0(off0))
            i += 1
          }
        }
        else {
          var i = 0
          while (i < noSamplesMax) {
            val off0 = indices.nextInt(length) * stride0
            fn(src0(off0))
            i += 1
          }
        }
      }
      else {
        if (stride0 == 1) {
          var i = 0
          while (i < noSamplesMax) {
            val off0 = offset0 + indices.nextInt(length)
            fn(src0(off0))
            i += 1
          }
        }
        else {
          var i = 0
          while (i < noSamplesMax) {
            val off0 = offset0 + indices.nextInt(length) * stride0
            fn(src0(off0))
            i += 1
          }
        }
      }
    }
    else {
      foreach(
        src0, offset0, stride0,
        length
      )(fn)
    }
  }

  @inline
  final def foreachActive[T](src0: SparseArray[T])
                            (fn: T => Unit)
  : Unit = {
    val data0  = src0.data
    val used0  = src0.activeSize
    var offset = 0
    while (offset < used0) {
      fn(data0(offset))
      offset += 1
    }
  }

  @inline
  final def foreachActivePair[T](src0: SparseArray[T])
                                (fn: (Int, T) => Unit)
  : Unit = {
    val indices = src0.index
    val data    = src0.data
    val used    = src0.activeSize
    var offset  = 0
    while (offset < used) {
      fn(indices(offset), data(offset))
      offset += 1
    }
  }

  @inline
  final def foreachEx[T](src0: SparseArray[T])
                        (fn0: T => Unit, fn1: => Unit)
  : Unit = {
    val indices0 = src0.index
    val data0    = src0.data
    val length0  = src0.size
    val used0    = src0.activeSize
    var i        = 0
    var offset   = 0
    while (offset < used0) {
      val index = indices0(offset)
      while (i < index) {
        fn1
        i += 1
      }

      fn0(data0(offset))
      offset += 1
      i      += 1
    }
    while (i < length0) {
      fn1
      i += 1
    }
  }

  @inline
  final def foreachEx[T, U](src0: Array[T], offset0: Int, stride0: Int,
                            src1: SparseArray[U])
                           (fn0: (T, U) => Unit, fn1: T => Unit)
  : Unit = {
    // Shorthands for frequently sued variables.
    val used1    = src1.activeSize
    val length1  = src1.size
    val indices1 = src1.index
    val data1    = src1.data

    // Process all pairs.
    var i    = 0
    var off0 = offset0
    var off1 = 0
    while (off1 < used1) {
      val index = indices1(off1)
      while (i < index) {
        fn1(src0(off0))
        off0 += stride0
        i    += 1
      }

      fn0(src0(off0), data1(off1))
      i    += 1
      off0 += stride0
      off1 += 1
    }

    while (i < length1) {
      fn1(src0(off0))
      off0 += stride0
      i    += 1
    }
  }

  @inline
  final def foreachPair[T](src0: Array[T])
                          (fn: (Int, T) => Unit)
  : Unit = foreachPair(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def foreachPair[T](src0:   Array[T], offset0: Int, stride0: Int,
                           length: Int)
                          (fn: (Int, T) => Unit)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          fn(i, src0(i))
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          fn(i, src0(i + offset0))
          i += 1
        }
      }
    }
    else {
      var off0 = 0
      var i    = 0
      while (i < length) {
        fn(i, src0(off0))
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def foreachPair[T, U](src0: Array[T],
                              src1: Array[U])
                             (fn: (Int, T, U) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    foreachPair(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreachPair[T, U](src0: Array[T], offset0: Int, stride0: Int,
                              src1: Array[U], offset1: Int, stride1: Int,
                              length: Int)
                             (fn: (Int, T, U) => Unit)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == 0 && offset1 == 0) {
        var i = 0
        while (i < length) {
          fn(
            i,
            src0(i),
            src1(i)
          )
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          fn(
            i,
            src0(i + offset0),
            src1(i + offset1)
          )
          i += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        fn(
          i,
          src0(off0),
          src1(off1)
        )
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def foreachPair[T, U](src0: Array[T], offset0: Int, stride0: Int,
                              src1: SparseArray[U])
                             (fn: (Int, T, U) => Unit)
  : Unit = {
    foreachPairEx(
      src0, offset0, stride0,
      src1
    )(fn, fn(_, _, src1.default))
  }

  @inline
  final def foreachPairEx[T, U](src0: Array[T], offset0: Int, stride0: Int,
                                src1: SparseArray[U])
                               (fn0: (Int, T, U) => Unit, fn1: (Int, T) => Unit)
  : Unit = {
    // Shorthands for frequently sued variables.
    val used1    = src1.activeSize
    val length1  = src1.size
    val indices1 = src1.index
    val data1    = src1.data

    // Process all pairs.
    var i    = 0
    var off0 = offset0
    var off1 = 0
    while (off1 < used1) {
      val index = indices1(off1)
      while (i < index) {
        fn1(i, src0(off0))
        off0 += stride0
        i    += 1
      }

      fn0(i, src0(off0), data1(off1))
      i    += 1
      off0 += stride0
      off1 += 1
    }

    while (i < length1) {
      fn1(i, src0(off0))
      off0 += stride0
      i    += 1
    }
  }

  @inline
  final def foreachParallel[T](src0: Array[T])
                              (fn: T => Unit)
  : Unit = foreachParallel(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def foreachParallel[T](src0: Array[T], offset0: Int, stride0: Int,
                               length: Int)
                              (fn: T => Unit)
  : Unit = {
    if (length == 1) {
      foreach(
        src0, offset0, stride0,
        length
      )(fn)
    }
    else {
      val tasks = map(
        src0, offset0, stride0,
        length
      )(v0 => Future(fn(v0)))
      finishAll(tasks)
    }
  }

  @inline
  final def foreachParallel[T](src0: Array[T], offset0: Int, stride0: Int,
                               length: Int, maxTaskLength: Int)
                              (fn: T => Unit)
  : Unit = {
    if (maxTaskLength < 1) {
      throw new IndexOutOfBoundsException
    }
    else if (maxTaskLength == 1) {
      foreachParallel(
        src0, offset0, stride0,
        length
      )(fn)
    }
    else {
      val noTasks = (length - 1 + maxTaskLength) / maxTaskLength
      if (noTasks == 1) {
        foreach(
          src0, offset0, stride0,
          length
        )(fn)
      }
      else {
        val tasks = tabulate(noTasks)(i => {
          val off1 = i * maxTaskLength
          val len0 = Math.max(maxTaskLength, length - off1)
          val off0 = offset0 + off1 * stride0
          Future(
            foreach(
              src0, off0, stride0,
              len0
            )(fn)
          )
        })
        finishAll(tasks)
      }
    }
  }

  @inline
  final def foreachParallelEx[T](src0: Array[T])
                                (fn: T => Unit)
  : Unit = foreachParallelEx(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def foreachParallelEx[T](src0: Array[T], offset0: Int, stride0: Int,
                                 length: Int)
                                (fn: T => Unit)
  : Unit = {
    val parallelism   = ForkJoinTasks.defaultForkJoinPool.getParallelism
    val maxTaskLength = (length - 1 + parallelism) / parallelism
    foreachParallel(
      src0, offset0, stride0,
      length, maxTaskLength
    )(fn)
  }

  @inline
  final def foreachParallel[T, U](src0: Array[T],
                                  src1: Array[U])
                                 (fn: (T, U) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    foreachParallel(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreachParallel[T, U](src0: Array[T], offset0: Int, stride0: Int,
                                  src1: Array[U], offset1: Int, stride1: Int,
                                  length: Int)
                                 (fn: (T, U) => Unit)
  : Unit = {
    val tasks = zip(
      src0, offset0, stride0,
      src1, offset1, stride1,
      length
    )((v0, v1) => Future(fn(v0, v1)))
    finishAll(tasks)
  }

  @inline
  final def foreachParallel[T, U, V](src0: Array[T],
                                     src1: Array[U],
                                     src2: Array[V])
                                    (fn: (T, U, V) => Unit)
  : Unit = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length
    )
    foreachParallel(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreachParallel[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                                     src1: Array[U], offset1: Int, stride1: Int,
                                     src2: Array[V], offset2: Int, stride2: Int,
                                     length: Int)
                                    (fn: (T, U, V) => Unit)
  : Unit = {
    val tasks = zip(
      src0, offset0, stride0,
      src1, offset1, stride1,
      src2, offset2, stride2,
      length
    )((v0, v1, v2) => Future(fn(v0, v1, v2)))
    finishAll(tasks)
  }

  @inline
  final def foreachParallel[T, U, V, W](src0: Array[T],
                                        src1: Array[U],
                                        src2: Array[V],
                                        src3: Array[W])
                                       (fn: (T, U, V, W) => Unit)
  : Unit = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length &&
      src0.length == src3.length
    )
    foreachParallel(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreachParallel[T, U, V, W](src0: Array[T], offset0: Int, stride0: Int,
                                        src1: Array[U], offset1: Int, stride1: Int,
                                        src2: Array[V], offset2: Int, stride2: Int,
                                        src3: Array[W], offset3: Int, stride3: Int,
                                        length: Int)
                                       (fn: (T, U, V, W) => Unit)
  : Unit = {
    val tasks = zip(
      src0, offset0, stride0,
      src1, offset1, stride1,
      src2, offset2, stride2,
      src3, offset3, stride3,
      length
    )((v0, v1, v2, v3) => Future(fn(v0, v1, v2, v3)))
    finishAll(tasks)
  }

  @inline
  final def foreachPairParallel[T](src0: Array[T])
                                  (fn: (Int, T) => Unit)
  : Unit = foreachPairParallel(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def foreachPairParallel[T](src0: Array[T], offset0: Int, stride0: Int,
                                   length: Int)
                                  (fn: (Int, T) => Unit)
  : Unit = {
    val tasks = mapPairs(
      src0, offset0, stride0,
      length
    )((i, v0) => Future(fn(i, v0)))
    finishAll(tasks)
  }

  @inline
  final def foreachPairParallel[T, U](src0: Array[T],
                                      src1: Array[U])
                                     (fn: (Int, T, U) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    foreachPairParallel(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreachPairParallel[T, U](src0: Array[T], offset0: Int, stride0: Int,
                                      src1: Array[U], offset1: Int, stride1: Int,
                                      length: Int)
                                     (fn: (Int, T, U) => Unit)
  : Unit = {
    val tasks = zipPairs(
      src0, offset0, stride0,
      src1, offset1, stride1,
      length
    )((i, v0, v1) => Future(fn(i, v0, v1)))
    finishAll(tasks)
  }

  @inline
  final def foreachPairParallel[T, U, V](src0: Array[T],
                                         src1: Array[U],
                                         src2: Array[V])
                                        (fn: (Int, T, U, V) => Unit)
  : Unit = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length
    )
    foreachPairParallel(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreachPairParallel[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                                         src1: Array[U], offset1: Int, stride1: Int,
                                         src2: Array[V], offset2: Int, stride2: Int,
                                         length: Int)
                                        (fn: (Int, T, U, V) => Unit)
  : Unit = {
    val tasks = zipPairs(
      src0, offset0, stride0,
      src1, offset1, stride1,
      src2, offset2, stride2,
      length
    )((i, v0, v1, v2) => Future(fn(i, v0, v1, v2)))
    finishAll(tasks)
  }

  @inline
  final def foreachPairParallel[T, U, V, W](src0: Array[T],
                                            src1: Array[U],
                                            src2: Array[V],
                                            src3: Array[W])
                                           (fn: (Int, T, U, V, W) => Unit)
  : Unit = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length &&
      src0.length == src3.length
    )
    foreachPairParallel(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def foreachPairParallel[T, U, V, W](src0: Array[T], offset0: Int, stride0: Int,
                                            src1: Array[U], offset1: Int, stride1: Int,
                                            src2: Array[V], offset2: Int, stride2: Int,
                                            src3: Array[W], offset3: Int, stride3: Int,
                                            length: Int)
                                           (fn: (Int, T, U, V, W) => Unit)
  : Unit = {
    val tasks = zipPairs(
      src0, offset0, stride0,
      src1, offset1, stride1,
      src2, offset2, stride2,
      src3, offset3, stride3,
      length
    )((i, v0, v1, v2, v3) => Future(fn(i, v0, v1, v2, v3)))
    finishAll(tasks)
  }

  @inline
  final def getAll[T](src0: Array[Future[T]])
                     (implicit tagT: ClassTag[T])
  : Array[T] = getAll(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def getAll[T](src0: Array[Future[T]], offset0: Int, stride0: Int,
                      length: Int)
                     (implicit tagT: ClassTag[T])
  : Array[T] = ArrayEx.map(
    src0, offset0, stride0,
    length
  )(FutureEx.get)

  @inline
  final def hashCode[T](src0: Array[T])
  : Int = hashCode(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def hashCode[T](src0: Array[T], offset0: Int, stride0: Int,
                        length: Int)
  : Int = foldLeft(
    hashSeed,
    src0, offset0, stride0,
    length
  )((tmp, value0) => MurmurHash3.mix(tmp, tmp.hashCode()))


  @inline
  final def interleave[T](src1: Array[T],
                          src2: Array[T])
                         (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src1.length + src2.length)
    interleave(
      result, 0, 1,
      src1, 0, 1,
      src1.length,
      src2, 0, 1,
      src2.length
    )
    result
  }

  @inline
  final def interleave[T](dst0: Array[T],
                          src1: Array[T],
                          src2: Array[T])
  : Unit = {
    require(
      dst0.length == src1.length * 2 &&
      dst0.length == src2.length * 2
    )
    interleave(
      dst0, 0, 1,
      src1, 0, 1,
      src1.length,
      src2, 0, 1,
      src2.length
    )
  }

  /**
    * Interleaving vertical concatenation.
    */
  @inline
  final def interleave[T](dst0: Array[T], offset0: Int, stride0: Int,
                          src1: Array[T], offset1: Int, stride1: Int,
                          length1: Int,
                          src2: Array[T], offset2: Int, stride2: Int,
                          length2: Int)
  : Unit = {
    set(
      dst0, offset0 + 0, stride0 * 2,
      src1, offset1,     stride1,
      length1
    )
    set(
      dst0, offset0 + 1, stride0 * 2,
      src2, offset2,     stride2,
      length2
    )
  }

  @inline
  final def interleave[T](dst0: Array[T],
                          src1: Array[T],
                          src2: Array[T],
                          src3: Array[T])
  : Unit = {
    require(
      dst0.length == src1.length * 3 &&
      dst0.length == src2.length * 3 &&
      dst0.length == src2.length * 3
    )
    interleave(
      dst0, 0, 1,
      src1, 0, 1,
      src1.length,
      src2, 0, 1,
      src2.length,
      src3, 0, 1,
      src3.length
    )
  }

  @inline
  final def interleave[T](dst0: Array[T], offset0: Int, stride0: Int,
                          src1: Array[T], offset1: Int, stride1: Int,
                          length1: Int,
                          src2: Array[T], offset2: Int, stride2: Int,
                          length2: Int,
                          src3: Array[T], offset3: Int, stride3: Int,
                          length3: Int)
  : Unit = {
    set(
      dst0, offset0 + 0, stride0 * 3,
      src1, offset1,     stride1,
      length1
    )
    set(
      dst0, offset0 + 1, stride0 * 3,
      src2, offset2,     stride2,
      length2
    )
    set(
      dst0, offset0 + 2, stride0 * 3,
      src3, offset3,     stride3,
      length3
    )
  }

  @inline
  final def interleave[T](dst0: Array[T], offset0: Int, stride0: Int,
                          src1: Array[T], offset1: Int, stride1: Int,
                          length1: Int,
                          src2: SparseArray[T])
  : Unit = {
    set(
      dst0, offset0 + 0, stride0 * 2,
      src1, offset1,     stride1,
      length1
    )
    setActive(
      dst0, offset0 + 1, stride0 * 2,
      src2
    )
  }

  @inline
  final def interleave[T](dst0: Array[T], offset0: Int, stride0: Int,
                          src1: SparseArray[T],
                          src2: Array[T], offset2: Int, stride2: Int,
                          length2: Int)
  : Unit = {
    setActive(
      dst0, offset0 + 0, stride0 * 2,
      src1
    )
    set(
      dst0, offset0 + 1, stride0 * 2,
      src2, offset2,     stride2,
      length2
    )
  }

  /**
    * Interleaving vertical concatenation.
    */
  @inline
  final def interleave[T](src0: SparseArray[T],
                          src1: SparseArray[T])
                         (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : SparseArray[T] = {
    // Create shorthands for frequently used variables.
    val length1  = src1.size
    val indices1 = src1.index
    val data1    = src1.data
    val used1    = src1.activeSize
    val length0  = src0.size
    val indices0 = src0.index
    val data0    = src0.data
    val used0    = src0.activeSize

    require(length0 == length1)

    // Allocate result buffer.
    val usedR    = used0 + used1
    val indicesR = new Array[Int](usedR)
    val dataR    = new Array[T](usedR)

    // Fill up result buffer until either this or other is finished.
    var iR = 0
    var i0 = 0
    var i1 = 0
    while (i0 < used0 && i1 < used1) {
      val index0 = indices0(i0)
      val index1 = indices1(i1)

      if (index0 <= index1) {
        indicesR(iR) = index0 + index0
        dataR(iR)    = data0(i0)
        iR += 1
        i0 += 1
      }
      if (index0 >= index1) {
        indicesR(iR) = index1 + index1 + 1
        dataR(iR)    = data1(i1)
        iR += 1
        i1 += 1
      }
    }

    // Remaining items in dv.
    while (i0 < used0) {
      val index    = indices0(i0)
      indicesR(iR) = index + index
      dataR(iR)    = data0(i0)
      iR += 1
      i0 += 1
    }

    // Remaining items in other.
    while (i1 < used1) {
      val index    = indices1(i1)
      indicesR(iR) = index + index + 1
      dataR(iR)    = data1(i1)
      iR += 1
      i1 += 1
    }

    assume(iR == usedR)
    new SparseArray(indicesR, dataR, usedR, length0 + length1, zeroT.zero)
  }

  @inline
  final def labelsToArray(noClasses: Int,
                          classNo:   Int)
  : Array[Real] = labelsToArray(noClasses, classNo, Real.one)

  @inline
  final def labelsToArray(noClasses: Int,
                          classNo:   Int,
                          value:     Real)
  : Array[Real] = {
    val result = new Array[Real](noClasses)
    result(classNo) = value
    result
  }

  @inline
  final def labelsToArrayEx(noClasses: Int,
                            classNos:  Seq[Int])
  : Array[Real] = labelsToArrayEx(noClasses, classNos, Real.one)

  @inline
  final def labelsToArrayEx(noClasses: Int,
                            classNos:  Seq[Int],
                            value:     Real)
  : Array[Real] = {
    val values = new Array[Real](noClasses)
    classNos.foreach(
      values(_) = value
    )
    values
  }

  @inline
  final def labelsToSparseArray(noClasses: Int,
                                classNo:   Int)
  : SparseArray[Real] = labelsToSparseArray(
    noClasses,
    classNo,
    Real.one
  )

  @inline
  final def labelsToSparseArray(noClasses: Int,
                                classNo:   Int,
                                value:     Real)
  : SparseArray[Real] = {
    require(classNo >= 0 && classNo < noClasses)
    val indices = Array(classNo)
    val data    = Array(value)
    new SparseArray(
      indices,
      data,
      1,
      noClasses,
      Real.zero
    )
  }

  @inline
  final def labelsToSparseArrayEx(noClasses: Int,
                                  classNos:  Seq[Int])
  : SparseArray[Real] = labelsToSparseArrayEx(noClasses, classNos, Real.one)

  @inline
  final def labelsToSparseArrayEx(noClasses: Int,
                                  classNos:  Seq[Int],
                                  value:     Real)
  : SparseArray[Real] = {
    require(classNos.forall(classNo => classNo >= 0 && classNo < noClasses))
    val indices = classNos.toArray
    val data    = fill(classNos.length, value)
    new SparseArray(
      indices,
      data,
      indices.length,
      noClasses,
      Real.zero
    )
  }

  /**
    * dst0 * (t - 1) + src1 * t
    */
  @inline
  final def lerp(dst0: Array[Real],
                 src1: Real,
                 t:    Real)
  : Unit = lerp(
    dst0, 0, 1,
    src1,
    dst0.length,
    t
  )

  /**
    * dst0 * (t - 1) + src1 * t
    */
  @inline
  final def lerp(dst0: Array[Real], offset0: Int, stride0: Int,
                 src1: Real,
                 length: Int,
                 t:      Real)
  : Unit = transform(
    dst0, offset0, stride0,
    length
  )(MathMacros.lerp(_, src1, t))

  /**
    * dst0 * (t - 1) + src1 * t
    */
  @inline
  final def lerp(dst0: Array[Real],
                 src1: Array[Real],
                 t:    Real)
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.lerp(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length,
      t
    )
  }

  /**
    * dst0 * (t - 1) + src1 * t
    */
  @inline
  final def lerp(dst0: Array[Real], offset0: Int, stride0: Int,
                 src1: Array[Real], offset1: Int, stride1: Int,
                 length: Int,
                 t:      Real)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = MathMacros.lerp(dst0(i), src1(i), t)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = MathMacros.lerp(dst0(off0), src1(off0), t)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = MathMacros.lerp(dst0(off0), src1(off1), t)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 < end0) {
        dst0(off0) = MathMacros.lerp(dst0(off0), src1(off1), t)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 * (t - 1) + src1 * src2 * t
    */
  @inline
  final def lerp(dst0: Array[Real],
                 src1: Array[Real],
                 src2: Array[Real],
                 t:    Real)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    ArrayEx.lerp(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length,
      t
    )
  }

  /**
    * dst0 * (t - 1) + src1 * src2 * t
    */
  @inline
  final def lerp(dst0: Array[Real], offset0: Int, stride0: Int,
                 src1: Array[Real], offset1: Int, stride1: Int,
                 src2: Array[Real], offset2: Int, stride2: Int,
                 length: Int,
                 t:      Real)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == offset1 && offset0 == offset2) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = MathMacros.lerp(dst0(i), src1(i) * src2(i), t)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = MathMacros.lerp(dst0(off0), src1(off0) * src2(off0), t)
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = MathMacros.lerp(dst0(off0), src1(off1) * src2(off2), t)
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 < end0) {
        dst0(off0) = MathMacros.lerp(dst0(off0), src1(off1) * src2(off2), t)
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def l1Norm(src0: Array[Real],
                   epsilon: Double)
  : Real = l1Norm(
    src0, 0, 1,
    src0.length,
    epsilon
  )

  @inline
  final def l1Norm(src0: Array[Real], offset0: Int, stride0: Int,
                   length0: Int,
                   epsilon: Double)
  : Real = {
    if (epsilon == 0.0) {
      foldLeft(
        Real.zero,
        src0, offset0, stride0,
        length0
      )((res, v0) => res + Math.abs(v0))
    }
    else {
      val result = foldLeft(
        0.0,
        src0, offset0, stride0,
        length0
      )((res, v0) => res + Math.sqrt(v0 * v0 + epsilon))
      Real(result)
    }
  }

  @inline
  final def l2Norm(src0: Array[Real],
                   epsilon: Double)
  : Real = Real(Math.sqrt(l2NormSq(src0) + epsilon))

  @inline
  final def l2Norm(src0: Array[Real], offset0: Int, stride0: Int,
                   length0: Int,
                   epsilon: Double)
  : Real = {
    val tmp = l2NormSq(
      src0, offset0, stride0,
      length0
    )
    Real(Math.sqrt(tmp + epsilon))
  }

  @inline
  final def l2NormSq(src0: Array[Real])
  : Real = l2NormSq(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def l2NormSq(src0: Array[Real], offset0: Int, stride0: Int,
                     length: Int)
  : Real = foldLeft(
    Real.zero,
    src0, offset0, stride0,
    length
  )(
    (res, v0) => res + v0 * v0
  )

  @inline
  final def map[T, U](src0: Array[T])
                     (fn: T => U)
                     (implicit tagU: ClassTag[U])
  : Array[U] = map(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def map[T, U](src0: Array[T], offset0: Int, stride0: Int,
                      length: Int)
                     (fn: T => U)
                     (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](length)
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(src0(i))
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(src0(i + offset0))
          i += 1
        }
      }
    }
    else {
      var off0 = offset0
      var i    = 0
      while (i < length) {
        result(i) = fn(src0(off0))
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def mapParallel[T, U](src0: Array[T])
                             (fn: T => U)
                             (implicit tagU: ClassTag[U])
  : Array[U] = mapParallel(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def mapParallel[T, U](src0: Array[T], offset0: Int, stride0: Int,
                              length: Int)
                             (fn: T => U)
                             (implicit tagU: ClassTag[U])
  : Array[U] = {
    if (length == 1) {
      map(
        src0, offset0, stride0,
        length
      )(fn)
    }
    else {
      val tasks = map(
        src0, offset0, stride0,
        length
      )(v0 => Future(fn(v0)))
      getAll(tasks)
    }
  }

  @inline
  final def mapParallel[T, U](src0: Array[T], offset0: Int, stride0: Int,
                              length: Int, maxTaskLength: Int)
                             (fn: T => U)
                             (implicit tagU: ClassTag[U])
  : Array[U] = {
    if (maxTaskLength < 1) {
      throw new IndexOutOfBoundsException
    }
    else if (maxTaskLength == 1) {
      mapParallel(
        src0, offset0, stride0,
        length
      )(fn)
    }
    else {
      val noTasks = (length - 1 + maxTaskLength) / maxTaskLength
      if (noTasks == 1) {
        map(
          src0, offset0, stride0,
          length
        )(fn)
      }
      else {
        val result = new Array[U](length)
        val tasks = tabulate(noTasks)(i => {
          val off1 = i * maxTaskLength
          val len0 = Math.max(maxTaskLength, length - off1)
          val off0 = offset0 + off1 * stride0
          Future(
            fill(
              result, off1, 1,
              src0,   off0, stride0,
              len0
            )(fn)
          )
        })
        finishAll(tasks)
        result
      }
    }
  }

  @inline
  final def mapParallelEx[T, U](src0: Array[T])
                               (fn: T => U)
                               (implicit tagU: ClassTag[U])
  : Array[U] = mapParallelEx(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def mapParallelEx[T, U](src0: Array[T], offset0: Int, stride0: Int,
                                length: Int)
                               (fn: T => U)
                               (implicit tagU: ClassTag[U])
  : Array[U] = {
    val parallelism   = ForkJoinTasks.defaultForkJoinPool.getParallelism
    val maxTaskLength = (length - 1 + parallelism) / parallelism
    mapParallel(
      src0, offset0, stride0,
      length, maxTaskLength
    )(fn)
  }

  @inline
  final def mapReduce[T, U](src0: Array[T])
                           (fnMap: T => U)
                           (fnReduce: (U, U) => U)
                           (implicit tagU: ClassTag[U])
  : U = mapReduce(
    src0, 0, 1,
    src0.length
  )(fnMap)(fnReduce)

  @inline
  final def mapReduce[T, U](src0: Array[T], offset0: Int, stride0: Int,
                            length: Int)
                           (fnMap: T => U)
                           (fnReduce: (U, U) => U)
                           (implicit tagU: ClassTag[U])
  : U = {
    val tmp = map(
      src0, offset0, stride0,
      length
    )(fnMap)
    reduceLeft(
      tmp
    )(fnReduce)
  }

  @inline
  final def mapActive[T, U](src0: SparseArray[T])
                           (fn: T => U)
                           (implicit tagU: ClassTag[U], zeroU: Zero[U])
  : SparseArray[U] = {
    val data0 = src0.data
    val used0 = src0.activeSize
    val data1 = new Array[U](used0)
    var i     = 0
    while (i < used0) {
      data1(i) = fn(data0(i))
      i += 1
    }
    new SparseArray(src0.index.clone, data1, used0, src0.size, zeroU.zero)
  }

  @inline
  final def mapPairs[T, U](src0: Array[T])
                          (fn: (Int, T) => U)
                          (implicit tagU: ClassTag[U])
  : Array[U] = mapPairs(
    src0, 0, 1,
    src0.length
  )(fn)

  @inline
  final def mapPairs[T, U](src0: Array[T], offset0: Int, stride0: Int,
                           length: Int)
                          (fn: (Int, T) => U)
                          (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](length)
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(i, src0(i))
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(i, src0(i + offset0))
          i += 1
        }
      }
    }
    else {
      var off0 = offset0
      var i    = 0
      while (i < result.length) {
        result(i) = fn(i, src0(off0))
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def max(src0: Array[Real])
  : Real = max(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def max(src0: Array[Real], offset0: Int, stride0: Int,
                length: Int)
  : Real = reduceLeft(
    src0, offset0, stride0,
    length
  )((res, src0) => if (src0 > res) src0 else res)

  @inline
  final def max(src0: SparseArray[Real])
  : Real = reduceLeft(
    src0
  )((res, src0) => if (src0 > res) src0 else res)

  @inline
  final def max(dst0: Array[Real],
                src1: Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    max(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def max(dst0: Array[Real], offset0: Int, stride0: Int,
                src1: Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )((dst0, src1) => if (dst0 > src1) dst0 else src1)

  @inline
  final def maxAbs(dst0: Array[Real], offset0: Int, stride0: Int,
                   length: Int)
  : Real = reduceLeft(
    dst0, offset0, stride0,
    length
  )((res, dst0) => {
    val tmp = Math.abs(dst0)
    if (tmp > res) tmp else res
  })

  @inline
  final def maxByAbs(dst0: Array[Real], offset0: Int, stride0: Int,
                     src1: Array[Real], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )((dst0, src1) => if (Math.abs(dst0) > Math.abs(src1)) dst0 else src1)

  @inline
  final def maxIndex(src0: Array[Real])
  : Int = maxIndex(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def maxIndex(src0:   Array[Real], offset0: Int, stride0: Int,
                     length: Int)
  : Int = {
    require(length > 0)
    var maxValue = src0(offset0)
    var maxIndex = 0

    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 1
        while (i < length) {
          val value = src0(i)
          if (value > maxValue) {
            maxValue = value
            maxIndex = i
          }
          i += 1
        }
      }
      else {
        var i = 1
        while (i < length) {
          val value = src0(i + offset0)
          if (value > maxValue) {
            maxValue = value
            maxIndex = i
          }
          i += 1
        }
      }
    }
    else {
      var off0 = offset0
      var i    = 1
      while (i < length) {
        val value = src0(off0)
        if (value > maxValue) {
          maxValue = value
          maxIndex = i
        }
        off0 += stride0
        i    += 1
      }
    }
    maxIndex
  }

  final val maxSize: Int = Int.MaxValue - 8

  @inline
  final def mean(src0: Array[Real])
  : Real = mean(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def mean(src0:   Array[Real], offset0: Int, stride0: Int,
                 length: Int)
  : Real = {
    val s = sum(
      src0, offset0, stride0,
      length
    )
    s / src0.length
  }

  @inline
  final def mean(src0: SparseArray[Real])
  : Real = sum(src0) / src0.length

  @inline
  def memoryUtilization[T](src0: SparseArray[T])
  : Real = src0.activeSize / Real(src0.size)

  @inline
  final def min(src0: Array[Real])
  : Real = min(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def min(src0: Array[Real], offset0: Int, stride0: Int,
                length: Int)
  : Real = reduceLeft(
    src0, offset0, stride0,
    length
  )((res, src0) => if (src0 < res) src0 else res)

  @inline
  final def min(src0: SparseArray[Real])
  : Real = reduceLeft(
    src0
  )((res, src0) => if (src0 < res) src0 else res)

  @inline
  final def min(dst0: Array[Real],
                src1: Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    min(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def min(dst0: Array[Real], offset0: Int, stride0: Int,
                src1: Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )((dst0, src1) => if (dst0 < src1) dst0 else src1)

  @inline
  final def minMax(src0: Array[Real])
  : RealRange = minMax(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def minMax(src0: Array[Real], offset0: Int, stride0: Int,
                   length: Int)
  : RealRange = {
    require(length > 0)
    var min0 = src0(offset0)
    var max0 = src0(offset0)

    foreach(
      src0, offset0 + stride0, stride0,
      length - 1
    )(src0 => {
      min0 = if (src0 < min0) src0 else min0
      max0 = if (src0 > max0) src0 else max0
    })
    RealRange(min0, max0)
  }

  @inline
  final def modulo(dst0: Array[Int],
                   src1: Array[Int])
  : Unit = {
    require(dst0.length == src1.length)
    modulo(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def modulo(dst0: Array[Int], offset0: Int, stride0: Int,
                   src1: Array[Int], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ % _)

  @inline
  final def multiply(dst0: Array[Real],
                     src1: Real)
  : Unit = multiply(
    dst0, 0, 1,
    src1,
    dst0.length
  )

  @inline
  final def multiply(dst0: Array[Real], offset0: Int, stride0: Int,
                     src1: Real,
                     length: Int)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          dst0(i) *= src1
          i       += 1
        }
      }
      else {
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) *= src1
          off0       += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) *= src1
        off0       += stride0
      }
    }
  }

  @inline
  final def multiply(dst0: Array[Int],
                     src1: Int)
  : Unit = multiply(
    dst0, 0, 1,
    src1,
    dst0.length
  )

  @inline
  final def multiply(dst0: Array[Int], offset0: Int, stride0: Int,
                     src1: Int,
                     length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    length
  )(_ * src1)

  @inline
  final def multiply(dst0: Array[Real],
                     src1: Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    multiply(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def multiply(dst0: Array[Real], offset0: Int, stride0: Int,
                     src1: Array[Real], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) *= src1(i)
            i       += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) *= src1(off0)
            off0       += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) *= src1(off1)
          off1       += 1
          off0       += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) *= src1(off1)
        off1       += stride1
        off0       += stride0
      }
    }
  }

  @inline
  final def multiply(dst0: Array[Real],
                     src1: Array[Real],
                     src2: Real,
                     length: Int)
  : Unit = {
    require(dst0.length == src1.length)
    multiply(
      dst0, 0, 1,
      src1, 0, 1,
      src2,
      dst0.length
    )
  }

  @inline
  final def multiply(dst0: Array[Real], offset0: Int, stride0: Int,
                     src1: Array[Real], offset1: Int, stride1: Int,
                     src2: Real,
                     length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) *= src1(i) * src2
            i       += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) *= src1(off0) * src2
            off0       += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) *= src1(off1) * src2
          off1       += 1
          off0       += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) *= src1(off1) * src2
        off1       += stride1
        off0       += stride0
      }
    }
  }

  @inline
  final def multiply(dst0: Array[Int],
                     src1: Array[Int])
  : Unit = {
    require(dst0.length == src1.length)
    multiply(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def multiply(dst0: Array[Int], offset0: Int, stride0: Int,
                     src1: Array[Int], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ * _)

  @inline
  final def pairsToMap[T](src0: Array[T])
  : SortedMap[Int, T] = {
    pairsToMap(
      src0, 0, 1,
      src0.length
    )
  }

  @inline
  final def pairsToMap[T](src0: Array[T], offset0: Int, stride0: Int,
                          length: Int)
  : SortedMap[Int, T] = {
    val builder = SortedMap.newBuilder[Int, T]
    builder.sizeHint(length)
    foreachPair(
      src0, offset0, stride0,
      length
    )(builder += Tuple2(_, _))
    builder.result()
  }

  @inline
  final def populationStdDev(src0: Array[Real],
                             epsilon: Double)
  : Real = {
    val sigmaSq = populationVariance(src0)
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def populationStdDev(src0: Array[Real], offset0: Int, stride0: Int,
                             length:  Int,
                             epsilon: Double)
  : Real = {
    val sigmaSq = populationVariance(
      src0, offset0, stride0,
      length
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def populationStdDev(src0: Array[Real],
                             mean1:   Real,
                             epsilon: Double)
  : Real = {
    val sigmaSq = populationVariance(
      src0,
      mean1
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def populationStdDev(src0:  Array[Real], offset0: Int, stride0: Int,
                             mean1:   Real,
                             length:  Int,
                             epsilon: Double)
  : Real = {
    val sigmaSq = populationVariance(
      src0, offset0, stride0,
      mean1,
      length
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def populationStdDev(src0: SparseArray[Real],
                             epsilon: Double)
  : Real = {
    val sigmaSq = populationVariance(src0)
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def populationStdDev(src0: SparseArray[Real],
                             mean1:   Real,
                             epsilon: Double)
  : Real = {
    val sigmaSq = populationVariance(
      src0,
      mean1
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def populationVariance(src0: Array[Real])
  : Real = populationVariance(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def populationVariance(src0: Array[Real], offset0: Int, stride0: Int,
                               length: Int)
  : Real = {
    val mu = mean(
      src0, offset0, stride0,
      length
    )
    populationVariance(
      src0, offset0, stride0,
      mu,
      length
    )
  }

  @inline
  final def populationVariance(src0: Array[Real],
                               mean1:  Real)
  : Real = populationVariance(
    src0, 0, 1,
    mean1,
    src0.length
  )

  @inline
  final def populationVariance(src0: Array[Real], offset0: Int, stride0: Int,
                               mean1:  Real,
                               length: Int)
  : Real = {
    val lengthSub1 = length - 1
    if (lengthSub1 > 0) {
      val diff = diffL2NormSq(
        src0, offset0, stride0,
        mean1,
        length
      )
      diff / lengthSub1
    }
    else {
      Real.zero
    }
  }

  @inline
  final def populationVariance(src0: SparseArray[Real])
  : Real = populationVariance(
    src0,
    mean(src0)
  )

  @inline
  final def populationVariance(src0: SparseArray[Real],
                               mean1: Real)
  : Real = {
    val length = src0.length
    if (length > 0) {
      val diff = diffL2NormSq(
        src0,
        mean1
      )
      diff / length
    }
    else {
      Real.zero
    }
  }

  @inline
  final def reduceLeft[T](src0: Array[T])
                         (fn: (T, T) => T)
  : T = {
    reduceLeft(
      src0, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def reduceLeft[T](src0: Array[T], offset0: Int, stride0: Int,
                          length: Int)
                         (fn: (T, T) => T)
  : T = {
    require(length > 0)
    var result = src0(offset0)

    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 1
        while (i < length) {
          result = fn(result, src0(i))
          i += 1
        }
      }
      else {
        val end0 = offset0 + length
        var off0 = offset0 + 1
        while (off0 < end0) {
          result = fn(result, src0(off0))
          off0 += 1
        }
      }
    }
    else {
      val end0 = offset0 + stride0 * length
      var off0 = offset0 + stride0
      while (off0 != end0) {
        result = fn(result, src0(off0))
        off0 += stride0
      }
    }

    result
  }

  @inline
  final def reduceLeft[T](src0: SparseArray[T])
                         (fn: (T, T) => T)
                         (implicit zeroT: Zero[T])
  : T = {
    val indices = src0.index
    val data    = src0.data
    val used    = src0.activeSize
    val length  = src0.size
    require(length > 0)

    var result = if (used == 0) zeroT.zero else data(0)
    var i      = 1
    var offset = if (used == 0) 0 else 1
    while (offset < used) {
      val index = indices(offset)
      if (i < index) {
        result = fn(result, zeroT.zero)
      }
      else {
        result = fn(result, data(offset))
        offset += 1
      }
      i += 1
    }
    while (i < length) {
      result = fn(result, zeroT.zero)
      i += 1
    }
    result
  }

  @inline
  final def reduceLeftActive[T](src0: SparseArray[T])
                               (fn: (T, T) => T)
  : T = {
    val data = src0.data
    val used = src0.activeSize
    require(used > 0)

    var result = data(0)
    var i      = 1
    while (i < used) {
      result = fn(result, data(i))
      i += 1
    }
    result
  }

  @inline
  final def reverse[T](dst0: Array[T])
  : Unit = {
    reverse(
      dst0, 0, 1,
      dst0.length
    )
  }

  @inline
  final def reverse[T](dst0: Array[T], offset0: Int, stride0: Int,
                       length: Int)
  : Unit = {
    var off0 = offset0
    var off1 = offset0 + stride0 * (length - 1)
    while (off0 < off1) {
      val tmp = dst0(off0)
      dst0(off0) = dst0(off1)
      dst0(off1) = tmp
      off1 -= stride0
      off0 += stride0
    }
  }

  @inline
  final def sampleStdDev(src0: Array[Real],
                         epsilon: Double)
  : Real = {
    val sigmaSq = sampleVariance(src0)
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def sampleStdDev(src0: Array[Real], offset0: Int, stride0: Int,
                         length:  Int,
                         epsilon: Double)
  : Real = {
    val sigmaSq = sampleVariance(
      src0, offset0, stride0,
      length
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def sampleStdDev(src0: Array[Real],
                         mean1:   Real,
                         epsilon: Double)
  : Real = {
    val sigmaSq = sampleVariance(
      src0,
      mean1
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }


  @inline
  final def sampleStdDev(src0: Array[Real], offset0: Int, stride0: Int,
                         mean1:   Real,
                         length:  Int,
                         epsilon: Double)
  : Real = {
    val sigmaSq = sampleVariance(
      src0, offset0, stride0,
      mean1,
      length
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def sampleStdDev(src0: SparseArray[Real],
                         epsilon: Double)
  : Real = {
    val sigmaSq = sampleVariance(src0)
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def sampleStdDev(src0: SparseArray[Real],
                         mean1:   Real,
                         epsilon: Double)
  : Real = {
    val sigmaSq = sampleVariance(
      src0,
      mean1
    )
    Real(Math.sqrt(sigmaSq + epsilon))
  }

  @inline
  final def sampleVariance(src0: Array[Real])
  : Real = sampleVariance(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def sampleVariance(src0: Array[Real], offset0: Int, stride0: Int,
                           length: Int)
  : Real = {
    val mu = mean(
      src0, offset0, stride0,
      length
    )
    sampleVariance(
      src0, offset0, stride0,
      mu,
      length
    )
  }

  @inline
  final def sampleVariance(src0: Array[Real],
                           mean1: Real)
  : Real = sampleVariance(
    src0, 0, 1,
    mean1,
    src0.length
  )

  @inline
  final def sampleVariance(src0: Array[Real], offset0: Int, stride0: Int,
                           mean1:  Real,
                           length: Int)
  : Real = {
    val lengthSub1 = src0.length - 1
    if (lengthSub1 > 0) {
      val diff = diffL2NormSq(
        src0, offset0, stride0,
        mean1,
        length
      )
      diff / lengthSub1
    }
    else {
      Real.zero
    }
  }

  @inline
  final def sampleVariance(src0: SparseArray[Real])
  : Real = sampleVariance(
    src0,
    mean(src0)
  )

  @inline
  final def sampleVariance(src0: SparseArray[Real],
                           mean1: Real)
  : Real = {
    val lengthSub1 = src0.length - 1
    if (lengthSub1 > 0) {
      val diff = diffL2NormSq(src0, mean1)
      diff / lengthSub1
    }
    else {
      Real.zero
    }
  }

  @inline
  def serialize(obj: JSerializable): Array[Byte] = {
    using(createOutputStream())(stream => {
      using(new ObjectOutputStream(stream))(stream => {
        stream.writeObject(obj)
      })
      stream.trim()
      stream.array
    })
  }

  /**
    * dst0 = src1
    */
  @inline
  final def set[T](dst0: Array[T], offset0: Int, stride0: Int,
                   src1: T,
                   length: Int)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var off0 = 0
        while (off0 < length) {
          dst0(off0) = src1
          off0 += 1
        }
      }
      else {
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = src1
          off0 += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = src1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = src1
    */
  @inline
  final def set[T](dst0: Array[T],
                   src1: Array[T])
  : Unit = {
    require(dst0.length == src1.length)
    set(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = src1
    */
  @inline
  final def set[T](dst0: Array[T], offset0: Int, stride0: Int,
                   src1: Array[T], offset1: Int, stride1: Int,
                   length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      Array.copy(
        src1, offset1,
        dst0, offset0,
        length
      )
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = src1
    */
  @inline
  final def set[T](dst0: Array[T],
                   src1: Array[T], range1: Range)
  : Unit = set(
    dst0, 0, 1,
    src1, range1
  )

  /**
    * dst0 = src1
    */
  @inline
  final def set[T](dst0: Array[T], offset0: Int, stride0: Int,
                   src1: Array[T], range1: Range)
  : Unit = {
    var off0 = offset0
    RangeEx.foreach(range1, off1 => {
      dst0(off0) = src1(off1)
      off0 += stride0
    })
  }


  /**
    * dst0 = src1
    */
  @inline
  final def setActive[T](dst0: Array[T], offset0: Int, stride0: Int,
                         src1: SparseArray[T])
  : Unit = {
    val used1    = src1.activeSize
    val indices1 = src1.index
    val data1    = src1.data
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < used1) {
          dst0(indices1(i)) = data1(i)
          i += 1
        }
      }
      else {
        var i = 0
        while (i < used1) {
          dst0(indices1(i) + offset0) = data1(i)
          i += 1
        }
      }
    }
    else if (stride0 == 2) {
      if (offset0 == 0) {
        var i = 0
        while (i < used1) {
          dst0(indices1(i) + indices1(i)) = data1(i)
          i += 1
        }
      }
      else {
        var i = 0
        while (i < used1) {
          dst0(indices1(i) + indices1(i) + offset0) = data1(i)
          i += 1
        }
      }
    }
    else {
      if (offset0 == 0) {
        var i = 0
        while (i < used1) {
          dst0(indices1(i) * stride0) = data1(i)
          i += 1
        }
      }
      else {
        var i = 0
        while (i < used1) {
          dst0(indices1(i) * stride0 + offset0) = data1(i)
          i += 1
        }
      }
    }
  }

  @inline
  final def set(dst0: Array[Real],
                beta: Real,
                src1: Array[Real])
  : Unit = {
    require(dst0.length == src1.length)
    set(
      dst0, 0, 1,
      beta,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def set(dst0: Array[Real], offset0: Int, stride0: Int,
                beta: Real,
                src1: Array[Real], offset1: Int, stride1: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = beta * src1(i)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = beta * src1(off0)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = beta * src1(off1)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = beta * src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = src1 * src2
    */
  @inline
  final def set(dst0: Array[Real],
                src1: Array[Real],
                src2: Array[Real],
                length: Int)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    set(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = src1 * src2
    */
  @inline
  final def set(dst0: Array[Real], offset0: Int, stride0: Int,
                src1: Array[Real], offset1: Int, stride1: Int,
                src2: Array[Real], offset2: Int, stride2: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == offset1 && offset0 == offset2) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = src1(i) * src2(i)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = src1(off0) * src2(off0)
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = src1(off1) * src2(off2)
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = src1(off1) * src2(off2)
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  /**
    * dst0 = beta * src1 * src2
    */
  @inline
  final def set(dst0: Array[Real],
                beta: Real,
                src1: Array[Real],
                src2: Array[Real],
                length: Int)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    set(
      dst0, 0, 1,
      beta,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )
  }

  /**
    * dst0 = beta * src1 * src2
    */
  @inline
  final def set(dst0: Array[Real], offset0: Int, stride0: Int,
                beta: Real,
                src1: Array[Real], offset1: Int, stride1: Int,
                src2: Array[Real], offset2: Int, stride2: Int,
                length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == offset1 && offset0 == offset2) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = beta * src1(i) * src2(i)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = beta * src1(off0) * src2(off0)
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = beta * src1(off1) * src2(off2)
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = beta * src1(off1) * src2(off2)
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }


  /**
    * Compute the sign. However, this works different from signum. It only
    * outputs -1 or 1
    */
  @inline
  final def sign(dst0: Array[Real])
  : Unit = sign(
    dst0, 0, 1,
    dst0.length
  )

  @inline
  final def sign(dst0: Array[Real], offset0: Int, stride0: Int,
                 length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    length
  )(dst0 => if (dst0 < Real.zero) -Real.one else Real.one)

  @inline
  final def sizeOf(src0: Array[Byte])
  : Long = 8L + src0.length

  @inline
  final def sizeOf(src0: Array[Short])
  : Long = 8L + src0.length * 2L

  @inline
  final def sizeOf(src0: Array[Int])
  : Long = 8L + src0.length * 4L

  @inline
  final def sizeOf(src0: Array[Long])
  : Long = 8L + src0.length * 8L

  @inline
  final def sizeOf(src0: Array[Float])
  : Long = 8L + src0.length * 4L

  @inline
  final def sizeOf(src0: Array[Double])
  : Long = 8L + src0.length * 8L

  @inline
  final def sizeOf(src0: SparseArray[Real])
  : Long = 8L + sizeOf(src0.index) + sizeOf(src0.data) + sparseArrayOverhead

  @inline
  final def skip[T](src0: Array[T], n: Int)
  : Array[T] = ArrayUtil.copyOfRange(src0, n, src0.length)

  @inline
  final def slice[T](src0: Array[T], from: Int, until: Int)
  : Array[T] = ArrayUtil.copyOfRange(src0, from, until)

  @inline
  final def slice[T](src0: Array[T], range: Range)
                    (implicit tag0: ClassTag[T])
  : Array[T] = {
    if (range.isInclusive) {
      if (range.step == 1) {
        slice(src0, range.start, range.end + 1)
      }
      else {
        val result = new Array[T](range.length)
        var j = range.start
        var i = 0
        while (i <= result.length) {
          result(i) = src0(j)
          j += range.step
          i += 1
        }
        result
      }
    }
    else {
      if (range.step == 1) {
        slice(src0, range.start, range.end)
      }
      else {
        val result = new Array[T](range.length)
        var j = range.start
        var i = 0
        while (i < result.length) {
          result(i) = src0(j)
          j += range.step
          i += 1
        }
        result
      }
    }
  }

  @inline
  final def sort(dst0: Array[Real])
  : Unit = java.util.Arrays.sort(dst0)

  @inline
  final def sort(dst0: Array[Real], offset0: Int,
                 length: Int)
  : Unit = java.util.Arrays.sort(dst0, offset0, offset0 + length)

  // used + size + default + lastReturnedPos
  final val sparseArrayOverhead
  : Long = 4L + 4L + 4L + 4L

  @inline
  final def split[T](src0: Array[T])
                    (implicit tagT: ClassTag[T])
  : Array[Array[T]] = split(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def split[T](src0: Array[T], offset0: Int, stride0: Int,
                     length: Int)
                    (implicit tagT: ClassTag[T])
  : Array[Array[T]] = map(
    src0, offset0, stride0,
    length
  )(Array(_))

  @inline
  final def sqr(dst0: Array[Real])
  : Unit = sqr(
    dst0, 0, 1,
    dst0.length
  )

  @inline
  final def sqr(dst0:   Array[Real], offset0: Int, stride0: Int,
                length: Int)
  : Unit = {
    transform(
      dst0, offset0, stride0,
      length
    )(x => x * x)
  }

  @inline
  final def sqrt(dst0: Array[Float])
  : Unit = sqrt(
    dst0, 0, 1,
    dst0.length
  )

  @inline
  final def sqrt(dst0:   Array[Float], offset0: Int, stride0: Int,
                 length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    length
  )(Math.sqrt(_).toFloat)

  @inline
  final def sqrt(dst0: Array[Double])
  : Unit = sqrt(
    dst0, 0, 1,
    dst0.length
  )

  @inline
  final def sqrt(dst0:   Array[Double], offset0: Int, stride0: Int,
                 length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    length
  )(x => Real(Math.sqrt(DoubleEx(x))))

  @inline
  final def subtract(src0: Real,
                     dst1: Array[Real])
  : Unit = subtract(
    src0,
    dst1, 0, 1,
    dst1.length
  )

  @inline
  final def subtract(src0:   Real,
                     dst1:   Array[Real], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = {
    if (src0 == Real.zero) {
      transform(
        dst1, offset1, stride1,
        length
      )(-_)
    }
    else {
      transform(
        dst1, offset1, stride1,
        length
      )(src0 - _)
    }
  }

  @inline
  final def subtract(dst0: Array[Float],
                     src1: Array[Float])
  : Unit = {
    require(dst0.length == src1.length)
    subtract(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def subtract(dst0:   Array[Float], offset0: Int, stride0: Int,
                     src1:   Array[Float], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) -= src1(i)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) -= src1(off0)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) -= src1(off1)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 < end0) {
        dst0(off0) -= src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def subtract(dst0: Array[Double],
                     src1: Array[Double])
  : Unit = {
    require(dst0.length == src1.length)
    subtract(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def subtract(dst0:   Array[Double], offset0: Int, stride0: Int,
                     src1:   Array[Double], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) -= src1(i)
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) -= src1(off0)
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) -= src1(off1)
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 < end0) {
        dst0(off0) -= src1(off1)
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def subtract(dst0: Array[Byte],
                     src1: Array[Byte])
  : Unit = {
    require(dst0.length == src1.length)
    subtract(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def subtract(dst0: Array[Byte], offset0: Int, stride0: Int,
                     src1: Array[Byte], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )((a, b) => (a.toInt - b.toInt).toByte)

  @inline
  final def subtract(dst0: Array[Int],
                     src1: Array[Int])
  : Unit = {
    require(dst0.length == src1.length)
    subtract(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )
  }

  @inline
  final def subtract(dst0: Array[Int], offset0: Int, stride0: Int,
                     src1: Array[Int], offset1: Int, stride1: Int,
                     length: Int)
  : Unit = transform(
    dst0, offset0, stride0,
    src1, offset1, stride1,
    length
  )(_ - _)

  @inline
  final def sum(src0: Array[Real])
  : Real = sum(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def sum(src0: Array[Real], offset0: Int, stride0: Int,
                length: Int)
  : Real = foldLeft(
    Real.zero,
    src0, offset0, stride0,
    length
  )(_ + _)

  @inline
  final def sum(src0: Array[Int], offset0: Int, stride0: Int,
                length: Int)
  : Int = foldLeft(
    0,
    src0, offset0, stride0,
    length
  )(_ + _)


  @inline
  final def sum(src0: Array[Long], offset0: Int, stride0: Int,
                length: Int)
  : Long = foldLeft(
    0L,
    src0, offset0, stride0,
    length
  )(_ + _)

  @inline
  final def sum(src0: SparseArray[Real])
  : Real = foldLeftActive(
    Real.zero,
    src0
  )(_ + _)

  @inline
  final def sumLong(src0: Array[Int], offset0: Int, stride0: Int,
                    length: Int)
  : Long = foldLeft(
    0L,
    src0, offset0, stride0,
    length
  )(_ + _)

  @inline
  final def tabulate[T](length: Int)
                       (fn: Int => T)
                       (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](length)
    tabulate(result)(fn)
    result
  }

  @inline
  final def tabulate[T](dst0: Array[T])
                       (fn: Int => T)
  : Unit = tabulate(
    dst0, 0, 1,
    dst0.length
  )(fn)

  @inline
  final def tabulate[T](dst0: Array[T], offset0: Int, stride0: Int,
                        length: Int)
                       (fn: Int => T)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          dst0(i) = fn(i)
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          dst0(i + offset0) = fn(i)
          i += 1
        }
      }
    }
    else {
      var off0 = offset0
      var i    = 0
      while (i < length) {
        dst0(off0) = fn(i)
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def tabulateParallel[T](length: Int)
                               (fn: Int => T)
                               (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = tabulate(
      length
    )(i => Future(fn(i)))
    getAll(tasks)
  }

  @inline
  final def tailor[T](array: SparseArray[T])
  : Unit = {
    val indices0 = array.index
    val data0    = array.data
    val used0    = array.activeSize
    if (indices0.length != used0 || data0.length != used0) {
      val indices1 = ArrayUtil.copyOf(indices0, used0)
      val data1    = ArrayUtil.copyOf(data0,    used0)
      array.use(indices1, data1, used0)
    }
  }

  @inline
  final def take[T](src0: Array[T], n: Int)
  : Array[T] = ArrayUtil.copyOf(src0, n)

  @inline
  final def takeRight[T](src0: Array[T], n: Int)
                        (implicit tag0: ClassTag[T])
  : Array[T] = ArrayUtil.copyOfRange(src0, src0.length - n, src0.length)

  @inline
  final def toArray[T](src0: SparseArray[T])
                      (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result  = new Array[T](src0.size)
    val indices = src0.index
    val data    = src0.data
    val used    = src0.activeSize
    var i       = 0
    while (i < used) {
      result(indices(i)) = data(i)
      i += 1
    }
    result
  }

  @inline
  final def toDataInputStream(src0: Array[Byte])
  : DataInputStream = toDataInputStream(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def toDataInputStream(src0: Array[Byte], offset0: Int, stride0: Int,
                              length: Int)
  : DataInputStream = {
    val stream = toInputStream(
      src0, offset0, stride0,
      length
    )
    new DataInputStream(stream)
  }

  @inline
  final def toInputStream(src0: Array[Byte])
  : FastByteArrayInputStream = toInputStream(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def toInputStream(src0: Array[Byte], offset0: Int, stride0: Int,
                          length: Int)
  : FastByteArrayInputStream = {
    if (stride0 == 1) {
      new FastByteArrayInputStream(src0, offset0, length)
    }
    else {
      logger.warn("Performance warning! Had to create a separate copy of the array.")
      val tmp = copy(
        src0, offset0, stride0,
        length
      )
      new FastByteArrayInputStream(tmp)
    }
  }

  @inline
  final def toObjectInputStream(src0: Array[Byte])
  : ObjectInputStream = toObjectInputStream(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def toObjectInputStream(src0: Array[Byte], offset0: Int, stride0: Int,
                                length: Int)
  : ObjectInputStream = {
    val stream = toInputStream(
      src0, offset0, stride0,
      length
    )
    new ObjectInputStream(stream)
  }

  @inline
  final def toSparseArray[T](src0: Array[T], offset0: Int, stride0: Int,
                             length: Int)
                            (predicate: T => Boolean)
                            (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : SparseArray[T] = {
    val indicesBuilder = Array.newBuilder[Int]
    val dataBuilder    = Array.newBuilder[T]

    // Process all pairs.
    foreachPair(src0, offset0, stride0, length)((index, value) => {
      if (predicate(value)) {
        indicesBuilder += index
        dataBuilder    += value
      }
    })

    val indices = indicesBuilder.result()
    val data    = dataBuilder.result()
    new SparseArray(indices, data, indices.length, src0.length, zeroT.zero)
  }

  @inline
  final def toString[T](src0: Array[T])
  : String = toString(
    src0, 0, 1,
    src0.length
  )

  @inline
  final def toString[T](src0: Array[T],
                        infix: String)
  : String = toString(
    src0, 0, 1,
    src0.length,
    infix
  )

  @inline
  final def toString[T](src0: Array[T], offset0: Int, stride0: Int,
                        length: Int)
  : String = toString(
    src0, offset0, stride0,
    length,
    ", "
  )

  @inline
  final def toString[T](src0: Array[T], offset0: Int, stride0: Int,
                        length: Int,
                        infix: String)
  : String = {
    val builder = StringBuilder.newBuilder
    foreach(
      src0, offset0, stride0,
      length
    )(src0 => builder ++= src0.toString ++= infix)

    // Remove last infix and return result
    if (builder.nonEmpty) {
      builder.length = builder.length - infix.length
    }
    builder.result()
  }

  @inline
  final def transform[T](dst0: Array[T])
                        (fn: T => T)
  : Unit = transform(
    dst0, 0, 1,
    dst0.length
  )(fn)

  @inline
  final def transform[T](dst0: Array[T], offset0: Int, stride0: Int,
                         length: Int)
                        (fn: T => T)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          dst0(i) = fn(dst0(i))
          i += 1
        }
      }
      else {
        var off = offset0
        val end = offset0 + length
        while (off < end) {
          dst0(off) = fn(dst0(off))
          off += 1
        }
      }
    }
    else {
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn(dst0(off0))
        off0 += stride0
      }
    }
  }

  @inline
  final def transform[T, U](dst0: Array[T],
                            src1: Array[U])
                           (fn: (T, U) => T)
  : Unit = {
    require(dst0.length == src1.length)
    transform(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U](dst0: Array[T], offset0: Int, stride0: Int,
                            src1: Array[U], offset1: Int, stride1: Int,
                            length: Int)
                           (fn: (T, U) => T)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(dst0(i), src1(i))
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = fn(dst0(off0), src1(off0))
            off0 += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = fn(dst0(off0), src1(off1))
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn(dst0(off0), src1(off1))
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def transform[T, U](dst0: Array[T],
                            src1: SparseArray[U])
                           (fn: (T, U) => T)
  : Unit = {
    transform(
      dst0, 0, 1,
      src1
    )(fn)
  }

  @inline
  final def transform[T, U](dst0: Array[T], offset0: Int, stride0: Int,
                            src1: SparseArray[U])
                           (fn: (T, U) => T)
  : Unit = {
    transformEx(
      dst0, offset0, stride0,
      src1
    )(fn, fn(_, src1.default))
  }

  @inline
  final def transform[T, U, V](dst0: Array[T],
                               src1: Array[U],
                               src2: Array[V])
                              (fn: (T, U, V) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    transform(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U, V](dst0: Array[T], offset0: Int, stride0: Int,
                               src1: Array[U], offset1: Int, stride1: Int,
                               src2: Array[V], offset2: Int, stride2: Int,
                               length: Int)
                              (fn: (T, U, V) => T)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (
        offset0 == offset1 &&
        offset0 == offset2
      ) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(dst0(i), src1(i), src2(i))
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = fn(dst0(off0), src1(off0), src2(off0))
            off0 += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = fn(dst0(off0), src1(off1), src2(off2))
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn(dst0(off0), src1(off1), src2(off2))
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def transform[T, U, V, W](dst0: Array[T],
                                  src1: Array[U],
                                  src2: Array[V],
                                  src3: Array[W])
                                 (fn: (T, U, V, W) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length &&
      dst0.length == src3.length
    )
    transform(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U, V, W](dst0: Array[T], offset0: Int, stride0: Int,
                                  src1: Array[U], offset1: Int, stride1: Int,
                                  src2: Array[V], offset2: Int, stride2: Int,
                                  src3: Array[W], offset3: Int, stride3: Int,
                                  length: Int)
                                 (fn: (T, U, V, W) => T)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1 && stride3 == 1) {
      if (
        offset0 == offset1 &&
        offset0 == offset2 &&
        offset0 == offset3
      ) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(dst0(i), src1(i), src2(i), src3(i))
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = fn(
              dst0(off0), src1(off0), src2(off0), src3(off0)
            )
            off0 += 1
          }
        }
      }
      else {
        var off3 = offset3
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = fn(
            dst0(off0), src1(off1), src2(off2), src3(off3)
          )
          off3 += 1
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off3 = offset3
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn(
          dst0(off0), src1(off1), src2(off2), src3(off3)
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def transform[T, U, V, W, X](dst0: Array[T],
                                     src1: Array[U],
                                     src2: Array[V],
                                     src3: Array[W],
                                     src4: Array[X])
                                    (fn: (T, U, V, W, X) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length &&
      dst0.length == src3.length &&
      dst0.length == src4.length
    )
    transform(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      src4, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U, V, W, X](dst0: Array[T], offset0: Int, stride0: Int,
                                     src1: Array[U], offset1: Int, stride1: Int,
                                     src2: Array[V], offset2: Int, stride2: Int,
                                     src3: Array[W], offset3: Int, stride3: Int,
                                     src4: Array[X], offset4: Int, stride4: Int,
                                     length: Int)
                                    (fn: (T, U, V, W, X) => T)
  : Unit = {
    if (
      stride0 == 1 && stride1 == 1 && stride2 == 1 &&
      stride3 == 1 && stride4 == 1
    ) {
      if (
        offset0 == offset1 &&
        offset0 == offset2 &&
        offset0 == offset3 &&
        offset0 == offset4
      ) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(
              dst0(i), src1(i), src2(i), src3(i), src4(i)
            )
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = fn(
              dst0(off0), src1(off0), src2(off0),
              src3(off0), src4(off0)
            )
            off0 += 1
          }
        }
      }
      else {
        var off4 = offset4
        var off3 = offset3
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = fn(
            dst0(off0), src1(off1), src2(off2),
            src3(off3), src4(off4)
          )
          off4 += 1
          off3 += 1
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off4 = offset4
      var off3 = offset3
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn(
          dst0(off0), src1(off1), src2(off2), src3(off3), src4(off4)
        )
        off4 += stride4
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def transform[T, U, V, W, X, Y](dst0: Array[T],
                                        src1: Array[U],
                                        src2: Array[V],
                                        src3: Array[W],
                                        src4: Array[X],
                                        src5: Array[Y])
                                       (fn: (T, U, V, W, X, Y) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length &&
      dst0.length == src3.length &&
      dst0.length == src4.length &&
      dst0.length == src5.length
    )
    transform(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      src4, 0, 1,
      src5, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U, V, W, X, Y](dst0: Array[T], offset0: Int, stride0: Int,
                                        src1: Array[U], offset1: Int, stride1: Int,
                                        src2: Array[V], offset2: Int, stride2: Int,
                                        src3: Array[W], offset3: Int, stride3: Int,
                                        src4: Array[X], offset4: Int, stride4: Int,
                                        src5: Array[Y], offset5: Int, stride5: Int,
                                        length: Int)
                                       (fn: (T, U, V, W, X, Y) => T)
  : Unit = {
    if (
      stride0 == 1 && stride1 == 1 && stride2 == 1 &&
      stride3 == 1 && stride4 == 1 && stride5 == 1
    ) {
      if (
        offset0 == offset1 &&
        offset0 == offset2 &&
        offset0 == offset3 &&
        offset0 == offset4 &&
        offset0 == offset5
      ) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(
              dst0(i), src1(i), src2(i),
              src3(i), src4(i), src5(i)
            )
            i += 1
          }
        }
        else {
          var off0 = offset0
          val end0 = offset0 + length
          while (off0 < end0) {
            dst0(off0) = fn(
              dst0(off0), src1(off0), src2(off0),
              src3(off0), src4(off0), src5(off0)
            )
            off0 += 1
          }
        }
      }
      else {
        var off5 = offset5
        var off4 = offset4
        var off3 = offset3
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        val end0 = offset0 + length
        while (off0 < end0) {
          dst0(off0) = fn(
            dst0(off0), src1(off1), src2(off2),
            src3(off3), src4(off4), src5(off5)
          )
          off5 += 1
          off4 += 1
          off3 += 1
          off2 += 1
          off1 += 1
          off0 += 1
        }
      }
    }
    else {
      var off5 = offset5
      var off4 = offset4
      var off3 = offset3
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      val end0 = offset0 + length * stride0
      while (off0 != end0) {
        dst0(off0) = fn(
          dst0(off0), src1(off1), src2(off2),
          src3(off3), src4(off4), src5(off5)
        )
        off5 += stride5
        off4 += stride4
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def transformActive[T](dst0: SparseArray[T])
                              (fn: T => T)
  : Unit = {
    val data0 = dst0.data
    val used0 = dst0.activeSize
    var off0  = 0
    while (off0 < used0) {
      data0(off0) = fn(data0(off0))
      off0 += 1
    }
  }

  @inline
  final def transformActive[T, U](dst0: Array[T],
                                  src1: SparseArray[U])
                                 (fn: (T, U) => T)
  : Unit = {
    require(dst0.length == src1.length)
    transformActive(
      dst0, 0, 1,
      src1
    )(fn)
  }

  @inline
  final def transformActive[T, U](dst0: Array[T], offset0: Int, stride0: Int,
                                  src1: SparseArray[U])
                                 (fn: (T, U) => T)
  : Unit = {
    val indices1 = src1.index
    val data1    = src1.data
    val used1    = src1.activeSize
    if (stride0 == 1) {
      var off1     = 0
      while (off1 < used1) {
        val off0 = offset0 + indices1(off1)
        dst0(off0) = fn(dst0(off0), data1(off1))
        off1 += 1
      }
    }
    else {
      var off1     = 0
      while (off1 < used1) {
        val off0 = offset0 + indices1(off1) * stride0
        dst0(off0) = fn(dst0(off0), data1(off1))
        off1 += 1
      }
    }
  }

  @inline
  final def transformEx[T, U](dst0: Array[T],
                              src1: SparseArray[U])
                             (fn0: (T, U) => T, fn1: T => T)
  : Unit = {
    require(dst0.length == src1.length)
    transformEx(
      dst0, 0, 1,
      src1
    )(fn0, fn1)
  }

  @inline
  final def transformEx[T, U](dst0: Array[T], offset0: Int, stride0: Int,
                              src1: SparseArray[U])
                             (fn0: (T, U) => T, fn1: T => T)
  : Unit = {
    // Shorthands for frequently sued variables.
    val length1  = src1.length
    val used1    = src1.activeSize
    val indices1 = src1.index
    val data1    = src1.data

    // Process all pairs.
    var i    = 0
    var off0 = offset0
    var off1 = 0
    while (off1 < used1) {
      val index = indices1(off1)
      while (i < index) {
        dst0(off0) = fn1(dst0(off0))
        off0 += 1
        i    += 1
      }

      dst0(off0) = fn0(dst0(off0), data1(off1))
      i    += 1
      off0 += 1
      off1 += 1
    }

    // If values remaining process them.
    while (i < length1) {
      dst0(off0) = fn1(dst0(off0))
      off0 += stride0
      i    += 1
    }
  }

  @inline
  final def transformPairs[T](dst0: Array[T])
                             (fn: (Int, T) => T)
  : Unit = transformPairs(
    dst0, 0, 1,
    dst0.length
  )(fn)

  @inline
  final def transformPairs[T](dst0: Array[T], offset0: Int, stride0: Int,
                              length: Int)
                             (fn: (Int, T) => T)
  : Unit = {
    if (stride0 == 1) {
      if (offset0 == 0) {
        var i = 0
        while (i < length) {
          dst0(i) = fn(i, dst0(i))
          i += 1
        }
      }
      else {
        var off0 = offset0
        var i    = 0
        while (i < length) {
          dst0(off0) = fn(i, dst0(off0))
          off0 += 1
          i    += 1
        }
      }
    }
    else {
      var off0 = offset0
      var i    = 0
      while (i < length) {
        dst0(off0) = fn(i, dst0(off0))
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def transformPairs[T, U](dst0: Array[T],
                                 src1: Array[U])
                                (fn: (Int, T, U) => T)
  : Unit = {
    require(dst0.length == src1.length)
    transformPairs(
      dst0, 0, 1,
      src1, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transformPairs[T, U](dst0: Array[T], offset0: Int, stride0: Int,
                                 src1: Array[U], offset1: Int, stride1: Int,
                                 length: Int)
                                (fn: (Int, T, U) => T)
  : Unit = {
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == offset1) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(i, dst0(i), src1(i))
            i += 1
          }
        }
        else {
          var off0 = offset0
          var i    = 0
          while (i < length) {
            dst0(off0) = fn(i, dst0(off0), src1(off0))
            off0 += 1
            i    += 1
          }
        }
      }
      else {
        var off1 = offset1
        var off0 = offset0
        var i    = 0
        while (i < length) {
          dst0(off0) = fn(i, dst0(off0), src1(off1))
          off1 += 1
          off0 += 1
          i    += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        dst0(off0) = fn(i, dst0(off0), src1(off1))
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def transformPairs[T, U, V](dst0: Array[T],
                                    src1: Array[U],
                                    src2: Array[V])
                                   (fn: (Int, T, U, V) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    transformPairs(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transformPairs[T, U, V](dst0: Array[T], offset0: Int, stride0: Int,
                                    src1: Array[U], offset1: Int, stride1: Int,
                                    src2: Array[V], offset2: Int, stride2: Int,
                                    length: Int)
                                   (fn: (Int, T, U, V) => T)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (
        offset0 == offset1 &&
        offset0 == offset2
      ) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(i, dst0(i), src1(i), src2(i))
            i += 1
          }
        }
        else {
          var off0 = offset0
          var i    = 0
          while (i < length) {
            dst0(off0) = fn(i, dst0(off0), src1(off0), src2(off0))
            off0 += 1
            i    += 1
          }
        }
      }
      else {
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        var i    = 0
        while (i < length) {
          dst0(off0) = fn(i, dst0(off0), src1(off1), src2(off2))
          off2 += 1
          off1 += 1
          off0 += 1
          i    += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        dst0(off0) = fn(i, dst0(off0), src1(off1), src2(off2))
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def transformPairs[T, U, V, W](dst0: Array[T],
                                       src1: Array[U],
                                       src2: Array[V],
                                       src3: Array[W])
                                      (fn: (Int, T, U, V, W) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    transformPairs(
      dst0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      dst0.length
    )(fn)
  }

  @inline
  final def transformPairs[T, U, V, W](dst0: Array[T], offset0: Int, stride0: Int,
                                       src1: Array[U], offset1: Int, stride1: Int,
                                       src2: Array[V], offset2: Int, stride2: Int,
                                       src3: Array[W], offset3: Int, stride3: Int,
                                       length: Int)
                                      (fn: (Int, T, U, V, W) => T)
  : Unit = {
    if (stride0 == 1 && stride1 == 1 && stride2 == 1 && stride3 == 1) {
      if (
        offset0 == offset1 &&
        offset0 == offset2 &&
        offset0 == offset3
      ) {
        if (offset0 == 0) {
          var i = 0
          while (i < length) {
            dst0(i) = fn(i, dst0(i), src1(i), src2(i), src3(i))
            i += 1
          }
        }
        else {
          var off0 = offset0
          var i    = 0
          while (i < length) {
            dst0(off0) = fn(
              i, dst0(off0), src1(off0), src2(off0), src3(off0)
            )
            off0 += 1
            i    += 1
          }
        }
      }
      else {
        var off3 = offset3
        var off2 = offset2
        var off1 = offset1
        var off0 = offset0
        var i    = 0
        while (i < length) {
          dst0(off0) = fn(
            i, dst0(off0), src1(off1), src2(off2), src3(off3)
          )
          off3 += 1
          off2 += 1
          off1 += 1
          off0 += 1
          i    += 1
        }
      }
    }
    else {
      var off3 = offset3
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        dst0(off0) = fn(
          i, dst0(off0), src1(off1), src2(off2), src3(off3)
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def utilization[T](src0: Array[T])
                          (predicate: T => Boolean)
  : Real = count(
    src0
  )(predicate) / Real(src0.length)

  @inline
  final def utilization[T](src0: SparseArray[T])
                          (predicate: T => Boolean)
  : Real = countActive(
    src0
  )(predicate) / Real(src0.size)

  @inline
  final def utilizationApprox[T](src0: Array[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Real = countApprox(
    src0,
    rng,
    noSamples
  )(predicate) / Real(src0.length)

  @inline
  final def utilizationApprox[T](src0: SparseArray[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Real = countActiveApprox(
    src0,
    rng,
    noSamples
  )(predicate) / Real(src0.size)

  @inline
  final def zip[T, U, V](src0: Array[T],
                         src1: Array[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    zip(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def zip[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                         src1: Array[U], offset1: Int, stride1: Int,
                         length: Int)
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = new Array[V](length)
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == 0 && offset1 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(
            src0(i),
            src1(i)
          )
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(
            src0(i + offset0),
            src1(i + offset1)
          )
          i += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        result(i) = fn(
          src0(off0),
          src1(off1)
        )
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def zip[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                         src1: SparseArray[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = zipEx(
    src0, offset0, stride0,
    src1
  )(fn, fn(_, src1.default))

  @inline
  final def zip[T, U, V](src0: SparseArray[T],
                         src1: SparseArray[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : SparseArray[V] = zipEx(
    src0,
    src1
  )(fn, fn(_, src1.default), fn(src0.default, _))

  @inline
  final def zip[T, U, V, W](src0: Array[T],
                            src1: Array[U],
                            src2: Array[V])
                           (fn: (T, U, V) => W)
                           (implicit tagW: ClassTag[W])
  : Array[W] = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length
    )
    zip(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def zip[T, U, V, W](src0: Array[T], offset0: Int, stride0: Int,
                            src1: Array[U], offset1: Int, stride1: Int,
                            src2: Array[V], offset2: Int, stride2: Int,
                            length: Int)
                           (fn: (T, U, V) => W)
                           (implicit tagW: ClassTag[W])
  : Array[W] = {
    val result = new Array[W](length)
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == 0 && offset1 == 0 && offset2 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(
            src0(i),
            src1(i),
            src2(i)
          )
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(
            src0(i + offset0),
            src1(i + offset1),
            src2(i + offset2)
          )
          i += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        result(i) = fn(
          src0(off0),
          src1(off1),
          src2(off2)
        )
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def zip[T, U, V, W, X](src0: Array[T],
                               src1: Array[U],
                               src2: Array[V],
                               src3: Array[W])
                              (fn: (T, U, V, W) => X)
                              (implicit tagW: ClassTag[X])
  : Array[X] = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length &&
      src0.length == src3.length
    )
    zip(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def zip[T, U, V, W, X](src0: Array[T], offset0: Int, stride0: Int,
                               src1: Array[U], offset1: Int, stride1: Int,
                               src2: Array[V], offset2: Int, stride2: Int,
                               src3: Array[W], offset3: Int, stride3: Int,
                               length: Int)
                              (fn: (T, U, V, W) => X)
                              (implicit tagW: ClassTag[X])
  : Array[X] = {
    val result = new Array[X](length)
    if (stride0 == 1 && stride1 == 1 && stride2 == 1 && stride3 == 1) {
      if (offset0 == 0 && offset1 == 0 && offset2 == 0 && offset3 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(
            src0(i),
            src1(i),
            src2(i),
            src3(i)
          )
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(
            src0(i + offset0),
            src1(i + offset1),
            src2(i + offset2),
            src3(i + offset3)
          )
          i += 1
        }
      }
    }
    else {
      var off3 = offset3
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        result(i) = fn(
          src0(off0),
          src1(off1),
          src2(off2),
          src3(off3)
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def zipEx[T, U, V](src0: Array[T],
                           src1: SparseArray[U])
                          (fn0: (T, U) => V, fn1: T => V)
                          (implicit tagV: ClassTag[V])
  : Array[V] = zipEx(
    src0, 0, 1,
    src1
  )(fn0, fn1)

  @inline
  final def zipEx[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                           src1: SparseArray[U])
                          (fn0: (T, U) => V, fn1: T => V)
                          (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    val result = new Array[V](src0.length)

    // Shorthands for frequently used variables.
    val used1    = src1.activeSize
    val indices1 = src1.index
    val data1    = src1.data

    // Process all pairs.
    var off1 = 0
    var off0 = offset0
    var i    = 0
    while (off1 < used1) {
      val index = indices1(off1)
      while (i < index) {
        result(i) = fn1(src0(off0))
        off0 += stride0
        i    += 1
      }

      result(i) = fn0(src0(off0), data1(off1))
      i    += 1
      off0 += 1
      off1 += 1
    }

    // If values remaining process them.
    while (i < result.length) {
      result(i) = fn1(src0(off0))
      off0 += stride0
      i    += 1
    }
    result
  }

  @inline
  final def zipEx[T, U, V](src0: SparseArray[T],
                           src1: SparseArray[U])
                          (fn0: (T, U) => V, fn1: T => V, fn2: U => V)
                          (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : SparseArray[V] = {
    // Shorthands for frequently sued variables.
    val indices1 = src1.index
    val data1    = src1.data
    val used1    = src1.activeSize
    val indices0 = src0.index
    val data0    = src0.data
    val used0    = src0.activeSize
    val length0  = src0.size

    require(length0 == src1.size)

    // Allocate result buffers.
    val indicesBuilder = Array.newBuilder[Int]
    indicesBuilder.sizeHint(length0)
    val dataBuilder = Array.newBuilder[V]
    dataBuilder.sizeHint(length0)

    // Process all pairs.
    var i0 = 0
    var i1 = 0
    while (i0 < used0 && i1 < used1) {
      val index0 = indices0(i0)
      val index1 = indices1(i1)

      if (index0 < index1) {
        indicesBuilder += index0
        dataBuilder    += fn1(data0(i0))
        i0             += 1
      }
      else if (index0 > index1) {
        indicesBuilder += index1
        dataBuilder    += fn2(data1(i1))
        i1             += 1
      }
      else {
        indicesBuilder += index0
        dataBuilder    += fn0(data0(i0), data1(i1))
        i0             += 1
        i1             += 1
      }
    }

    while (i0 < used0) {
      indicesBuilder += indices0(i0)
      dataBuilder    += fn1(data0(i0))
      i0             += 1
    }

    while (i1 < used1) {
      indicesBuilder += indices1(i1)
      dataBuilder    += fn2(data1(i1))
      i1             += 1
    }

    val indices = indicesBuilder.result()
    val data    = dataBuilder.result()
    new SparseArray(indices, data, data.length, length0, zeroV.zero)
  }

  @inline
  final def zipPairs[T, U, V](src0: Array[T],
                              src1: Array[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    zipPairs(
      src0, 0, 1,
      src1, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def zipPairs[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                              src1: Array[U], offset1: Int, stride1: Int,
                              length: Int)
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = new Array[V](length)
    if (stride0 == 1 && stride1 == 1) {
      if (offset0 == 0 && offset1 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(
            i,
            src0(i),
            src1(i)
          )
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(
            i,
            src0(i + offset0),
            src1(i + offset1)
          )
          i += 1
        }
      }
    }
    else {
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        result(i) = fn(
          i,
          src0(off0),
          src1(off1)
        )
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def zipPairs[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                              src1: SparseArray[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = zipPairsEx(
    src0, offset0, stride0,
    src1
  )(fn, fn(_, _, src1.default))

  @inline
  final def zipPairs[T, U, V, W](src0: Array[T],
                                 src1: Array[U],
                                 src2: Array[V])
                                (fn: (Int, T, U, V) => W)
                                (implicit tagW: ClassTag[W])
  : Array[W] = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length
    )
    zipPairs(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def zipPairs[T, U, V, W](src0: Array[T], offset0: Int, stride0: Int,
                                 src1: Array[U], offset1: Int, stride1: Int,
                                 src2: Array[V], offset2: Int, stride2: Int,
                                 length: Int)
                                (fn: (Int, T, U, V) => W)
                                (implicit tagW: ClassTag[W])
  : Array[W] = {
    val result = new Array[W](length)
    if (stride0 == 1 && stride1 == 1 && stride2 == 1) {
      if (offset0 == 0 && offset1 == 0 && offset2 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(
            i,
            src0(i),
            src1(i),
            src2(i)
          )
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(
            i,
            src0(i + offset0),
            src1(i + offset1),
            src2(i + offset2)
          )
          i += 1
        }
      }
    }
    else {
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        result(i) = fn(i, src0(off0), src1(off1), src2(off2))
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def zipPairs[T, U, V, W, X](src0: Array[T],
                                    src1: Array[U],
                                    src2: Array[V],
                                    src3: Array[W])
                                   (fn: (Int, T, U, V, W) => X)
                                   (implicit tagW: ClassTag[X])
  : Array[X] = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length &&
      src0.length == src3.length
    )
    zipPairs(
      src0, 0, 1,
      src1, 0, 1,
      src2, 0, 1,
      src3, 0, 1,
      src0.length
    )(fn)
  }

  @inline
  final def zipPairs[T, U, V, W, X](src0: Array[T], offset0: Int, stride0: Int,
                                    src1: Array[U], offset1: Int, stride1: Int,
                                    src2: Array[V], offset2: Int, stride2: Int,
                                    src3: Array[W], offset3: Int, stride3: Int,
                                    length: Int)
                                   (fn: (Int, T, U, V, W) => X)
                                   (implicit tagW: ClassTag[X])
  : Array[X] = {
    val result = new Array[X](length)
    if (stride0 == 1 && stride1 == 1 && stride2 == 1 && stride3 == 1) {
      if (offset0 == 0 && offset1 == 0 && offset2 == 0 && offset3 == 0) {
        var i = 0
        while (i < length) {
          result(i) = fn(
            i,
            src0(i),
            src1(i),
            src2(i),
            src3(i)
          )
          i += 1
        }
      }
      else {
        var i = 0
        while (i < length) {
          result(i) = fn(
            i,
            src0(i + offset0),
            src1(i + offset1),
            src2(i + offset2),
            src3(i + offset3)
          )
          i += 1
        }
      }
    }
    else {
      var off3 = offset3
      var off2 = offset2
      var off1 = offset1
      var off0 = offset0
      var i    = 0
      while (i < length) {
        result(i) = fn(
          i,
          src0(off0),
          src1(off1),
          src2(off2),
          src3(off3)
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    result
  }

  @inline
  final def zipPairsEx[T, U, V](src0: Array[T], offset0: Int, stride0: Int,
                                src1: SparseArray[U])
                               (fn0: (Int, T, U) => V, fn1: (Int, T) => V)
                               (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    val result = new Array[V](src0.length)

    val used1    = src1.activeSize
    val data1    = src1.data
    val indices1 = src1.index

    // Process all pairs.
    var i    = 0
    var off0 = offset0
    var off1 = 0
    while (off1 < used1) {
      val index = indices1(off1)
      while (i < index) {
        result(i) = fn1(i, src0(off0))
        off0 += stride0
        i    += 1
      }

      result(i) = fn0(i, src0(off0), data1(off1))
      i    += 1
      off0 += stride0
      off1 += 1
    }

    // If values remaining process them.
    while (i < result.length) {
      result(i) = fn1(i, src0(off0))
      off0 += stride0
      i    += 1
    }

    result
  }

  @inline
  final def zipPairsEx[T, U, V](src0: SparseArray[T],
                                src1: SparseArray[U])
                               (fn0: (Int, T, U) => V,
                                fn1: (Int, T) => V,
                                fn2: (Int, U) => V)
                               (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : SparseArray[V] = {
    // Shorthands for frequently sued variables.
    val indices1 = src1.index
    val data1    = src1.data
    val used1    = src1.activeSize
    val indices0 = src0.index
    val data0    = src0.data
    val used0    = src0.activeSize
    val length0  = src0.size

    require(length0 == src1.size)

    // Allocate result buffer.
    val indicesBuilder = Array.newBuilder[Int]
    indicesBuilder.sizeHint(length0)
    val dataBuilder = Array.newBuilder[V]
    dataBuilder.sizeHint(length0)

    // Process all pairs.
    var i0 = 0
    var i1 = 0
    while (i0 < used0 && i1 < used1) {
      val index0 = indices0(i0)
      val index1 = indices1(i1)

      if (index0 < index1) {
        indicesBuilder += index0
        dataBuilder    += fn1(index0, data0(i0))
        i0             += 1
      }
      else if (index0 > index1) {
        indicesBuilder += index1
        dataBuilder    += fn2(index1, data1(i1))
        i1             += 1
      }
      else {
        indicesBuilder += index0
        dataBuilder    += fn0(index0, data0(i0), data1(i1))
        i0             += 1
        i1             += 1
      }
    }

    while (i0 < used0) {
      val index0 = indices0(i0)
      indicesBuilder += index0
      dataBuilder    += fn1(index0, data0(i0))
      i0             += 1
    }

    while (i1 < used1) {
      val index1 = indices1(i1)
      indicesBuilder += index1
      dataBuilder    += fn2(index1, data1(i1))
      i1             += 1
    }

    val indices = indicesBuilder.result()
    val data    = dataBuilder.result()
    new SparseArray(indices, data, data.length, length0, zeroV.zero)
  }

}
