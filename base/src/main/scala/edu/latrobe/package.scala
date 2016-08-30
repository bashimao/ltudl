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

package edu

import breeze.stats.distributions.Rand
import java.nio._
import java.util.concurrent._
import org.slf4j._
import scala.language.implicitConversions

/**
  * M_LANGER: Do not forget to compile with -Xdisable-assertions for production
  * builds!
  */
package object latrobe {

  final val logger
  : Logger = LoggerFactory.getLogger("edu.latrobe")
  logger.info("org.latrobe.logger enabled!")

  final val hashSeed
  : Int = "latrobe".hashCode

  // Following the idea from http://blog.omega-prime.co.uk/?p=17, but using our fancy disposable interface.
  @inline
  final def using[T0 <: AutoCloseable, U](v0: T0)
                                         (fn: T0 => U)
  : U = {
    try {
      fn(v0)
    }
    catch {
      case e: Exception =>
        logger.error(s"using: ", e)
        throw e
    }
    finally {
      if (v0 != null) {
        v0.close()
      }
    }
  }

  @inline
  final def using[T0 <: AutoCloseable, T1 <: AutoCloseable, U](v0: T0, v1: T1)
                                                              (fn: (T0, T1) => U)
  : U = {
    try {
      fn(v0, v1)
    }
    catch {
      case e: Exception =>
        logger.error(s"using: ", e)
        throw e
    }
    finally {
      if (v1 != null) {
        v1.close()
      }
      if (v0 != null) {
        v0.close()
      }
    }
  }

  @inline
  final def using[T0 <: AutoCloseable, T1 <: AutoCloseable, T2 <: AutoCloseable, U](v0: T0, v1: T1, v2: T2)
                                                                                   (fn: (T0, T1, T2) => U)
  : U = {
    try {
      fn(v0, v1, v2)
    }
    catch {
      case e: Exception =>
        logger.error(s"using: ", e)
        throw e
    }
    finally {
      if (v2 != null) {
        v2.close()
      }
      if (v1 != null) {
        v1.close()
      }
      if (v0 != null) {
        v0.close()
      }
    }
  }

  @inline
  final def using[T0 <: AutoCloseable, T1 <: AutoCloseable, T2 <: AutoCloseable, T3 <: AutoCloseable, U](v0: T0, v1: T1, v2: T2, v3: T3)
                                                                                                        (fn: (T0, T1, T2, T3) => U)
  : U = {
    try {
      fn(v0, v1, v2, v3)
    }
    catch {
      case e: Exception =>
        logger.error(s"using: ", e)
        throw e
    }
    finally {
      if (v3 != null) {
        v3.close()
      }
      if (v2 != null) {
        v2.close()
      }
      if (v1 != null) {
        v1.close()
      }
      if (v0 != null) {
        v0.close()
      }
    }
  }

  @inline
  final def using[T0 <: AutoCloseable, T1 <: AutoCloseable, T2 <: AutoCloseable, T3 <: AutoCloseable, T4 <: AutoCloseable, U](v0: T0, v1: T1, v2: T2, v3: T3, v4: T4)
                                                                                                                             (fn: (T0, T1, T2, T3, T4) => U)
  : U = {
    try {
      fn(v0, v1, v2, v3, v4)
    }
    catch {
      case e: Exception =>
        logger.error(s"using: ", e)
        throw e
    }
    finally {
      if (v4 != null) {
        v4.close()
      }
      if (v3 != null) {
        v3.close()
      }
      if (v2 != null) {
        v2.close()
      }
      if (v1 != null) {
        v1.close()
      }
      if (v0 != null) {
        v0.close()
      }
    }
  }

  @inline
  final def using[T0 <: AutoCloseable, T1 <: AutoCloseable, T2 <: AutoCloseable, T3 <: AutoCloseable, T4 <: AutoCloseable, T5 <: AutoCloseable, U](v0: T0, v1: T1, v2: T2, v3: T3, v4: T4, v5: T5)
                                                                                                                                                  (fn: (T0, T1, T2, T3, T4, T5) => U)
  : U = {
    try {
      fn(v0, v1, v2, v3, v4, v5)
    }
    catch {
      case e: Exception =>
        logger.error(s"using: ", e)
        throw e
    }
    finally {
      if (v5 != null) {
        v5.close()
      }
      if (v4 != null) {
        v4.close()
      }
      if (v3 != null) {
        v3.close()
      }
      if (v2 != null) {
        v2.close()
      }
      if (v1 != null) {
        v1.close()
      }
      if (v0 != null) {
        v0.close()
      }
    }
  }

  @inline
  final def using[T0 <: AutoCloseable, U](a0: Array[T0])
                                         (fn: (Array[T0]) => U)
  : U = {
    try {
      fn(a0)
    }
    catch {
      case e: Exception =>
        logger.error(s"using: ", e)
        throw e
    }
    finally {
      ArrayEx.foreach(
        a0
      )(a0 => {
        if (a0 != null) {
          a0.close()
        }
      })
    }
  }

  @inline
  final def locking[T](semaphore: Semaphore, noPermits: Int = 1)
                      (block: => T)
  : T = {
    try {
      semaphore.acquire(noPermits)
      block
    }
    catch {
      case e: Exception =>
        logger.error(s"locking: ", e)
        throw e
    }
    finally {
      semaphore.release(noPermits)
    }
  }

  object Implicits {

    final implicit def distributionToRand[T](distribution: Distribution[T])
    : Rand[T] = {
      new Rand[T] {
        override def draw()
        : T = distribution.sample()
      }
    }

  }

  final implicit class IntFunctions(i: Int) {

    @inline
    def ?+(other: Int): Int = {
      val result = i.toLong + other.toLong
      if (result < Int.MinValue || result > Int.MaxValue) {
        throw new ArithmeticException("Integer overflow!")
      }
      result.toInt
    }

    @inline
    def ?-(other: Int): Int = {
      val result = i.toLong - other.toLong
      if (result < Int.MinValue || result > Int.MaxValue) {
        throw new ArithmeticException("Integer overflow!")
      }
      result.toInt
    }

    @inline
    def ?*(other: Int): Int = {
      val result = i.toLong * other.toLong
      if (result < Int.MinValue || result > Int.MaxValue) {
        throw new ArithmeticException("Integer overflow!")
      }
      result.toInt
    }

    @inline
    def !+(other: Int): Boolean = {
      val result = i.toLong + other.toLong
      result >= Int.MinValue && result <= Int.MaxValue
    }

    @inline
    def !-(other: Int): Boolean = {
      val result = i.toLong - other.toLong
      result >= Int.MinValue && result <= Int.MaxValue
    }

    @inline
    def !*(other: Int): Boolean = {
      val result = i.toLong * other.toLong
      result >= Int.MinValue && result <= Int.MaxValue
    }

  }

  // ---------------------------------------------------------------------------
  //    REAL SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  /**
   * If you change this type, you also have to touch Real.scala!
   */
  type Real = Double

  val Real = DoubleEx

  type NativeRealBuffer = DoubleBuffer
  */
  // ---------------------------------------------------------------------------
  //    REAL SWITCH FLOAT
  // ---------------------------------------------------------------------------
  ///*
  /**
    * If you change this type, you also have to touch Real.scala!
    */
  type Real = Float

  val Real = FloatEx

  type NativeRealBuffer = FloatBuffer
  //*/
  // ---------------------------------------------------------------------------
  //    REAL SWITCH END
  // ---------------------------------------------------------------------------

  /*
  final implicit class IterableFunctions[T](it: Iterable[T]) {

    @inline
    def fastFoldLeft[U](fnHead: T => U, fnTail: (U, T) => U): U = {
      val iter = it.iterator
      var z = fnHead(iter.next())
      while (iter.hasNext) {
        z = fnTail(z, iter.next())
      }
      z
    }

  }

  final implicit class ArrayFunctions[T](arr: Array[T]) {

    @inline
    def fastZipEx[U, V: ClassTag](other: Array[U])
                                 (fn0: (T, U) => V, fn1: T => V, fn2: U => V)
    : Array[V] = {
      if (arr.length == other.length) {
        fastZip(other)(fn0)
      }
      else {
        val result = Array.ofDim[V](Math.max(arr.length, other.length))

        var i = 0
        while (i < arr.length && i < other.length) {
          result(i) = fn0(arr(i), other(i))
          i += 1
        }

        while (i < arr.length) {
          result(i) = fn1(arr(i))
          i += 1
        }

        while (i < other.length) {
          result(i) = fn2(other(i))
          i += 1
        }

        result
      }
    }
  }

  final implicit class IntArrayFunctions(arr: Array[Int]) {

    @inline
    def sizeInByte: Long = 8L + arr.length * 4L

  }
  */

  type JSerializable = java.io.Serializable

  final val LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE",
    1024 * 1024,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ABS
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ABS",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DOT
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DOT",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_FILL
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_FILL",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L1_NORM
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L1_NORM",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L2_NORM
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L2_NORM",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_LERP
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_LERP",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MIN
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MIN",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SET
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SET",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SIGN
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SIGN",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQR
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQR",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQRT
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQRT",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SUM
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SUM",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TABULATE
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TABULATE",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TRANSFORM
  : Int = Environment.parseInt(
    "LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TRANSFORM",
    LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE,
    _ > 0
  )

  final val LTU_REDUNDANT_CALL_TO_CLOSE_WARNING
  : Boolean = Environment.parseBoolean(
    "LTU_REDUNDANT_CALL_TO_CLOSE_WARNING",
    default = false
  )


}
