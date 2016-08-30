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

import scala.concurrent._
import scala.reflect._
import scala.concurrent.ExecutionContext.Implicits.global

object RangeEx {

  @inline
  final def clip(range: Range, value: Int): Int = {
    val min = range.min
    if (value < min) {
      return min
    }
    val max = range.max
    if (value > max) {
      return max
    }
    value
  }

  @inline
  final def foreach(from0: Int, until0: Int,
                    fn:    Int => Unit)
  : Unit = {
    var off0 = from0
    while (off0 < until0) {
      fn(off0)
      off0 += 1
    }
  }

  @inline
  final def foreach(from0: Int, until0: Int, stride0: Int,
                    fn:    Int => Unit)
  : Unit = {
    var off0 = from0
    val end0 = from0 + stride0 * noStepsFor(from0, until0, stride0)
    while (off0 != end0) {
      fn(off0)
      off0 += stride0
    }
  }

  @inline
  final def foreach(range0: Range,
                    fn:     Int => Unit)
  : Unit = {
    val stride0 = range0.step
    var off0    = range0.start
    val end0    = range0.terminalElement

    while (off0 != end0) {
      fn(off0)
      off0 += stride0
    }
  }

  @inline
  final def foreach(range0: Range,
                    range1: Range,
                    fn:     (Int, Int) => Unit)
  : Unit = {
    val stride1 = range1.step
    var off1    = range1.start
    val length1 = range1.length
    val stride0 = range0.step
    var off0    = range0.start
    val length0 = range0.length

    require(length0 == length1)

    val end0 = range0.terminalElement
    while (off0 != end0) {
      fn(off0, off1)
      off1 += stride1
      off0 += stride0
    }
  }

  @inline
  final def foreach(range0: Range,
                    range1: Range,
                    range2: Range,
                    fn:     (Int, Int, Int) => Unit)
  : Unit = {
    val stride2 = range2.step
    var off2    = range2.start
    val length2 = range2.length
    val stride1 = range1.step
    var off1    = range1.start
    val length1 = range1.length
    val stride0 = range0.step
    var off0    = range0.start
    val length0 = range0.length

    require(
      length0 == length1 &&
      length0 == length2
    )

    val end0 = range0.terminalElement
    while (off0 != end0) {
      fn(off0, off1, off2)
      off2 += stride2
      off1 += stride1
      off0 += stride0
    }
  }

  @inline
  final def foreachParallel(from0: Int, until0: Int,
                            fn:    Int => Unit)
  : Unit = {
    val tasks = map(
      from0, until0
    )(off0 => Future(fn(off0)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachParallel(from0: Int, until0: Int, stride0: Int,
                            fn:    Int => Unit)
  : Unit = {
    val tasks = map(
      from0, until0, stride0
    )(off0 => Future(fn(off0)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachParallel(range0: Range,
                            fn:     Int => Unit)
  : Unit = {
    val tasks = map(
      range0
    )(off0 => Future(fn(off0)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachParallel(range0: Range,
                            range1: Range,
                            fn:     (Int, Int) => Unit)
  : Unit = {
    val tasks = map(
      range0,
      range1
    )((off0, off1) => Future(fn(off0, off1)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachParallel(range0: Range,
                            range1: Range,
                            range2: Range,
                            fn:     (Int, Int, Int) => Unit)
  : Unit = {
    val tasks = map(
      range0,
      range1,
      range2
    )((off0, off1, off2) => Future(fn(off0, off1, off2)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachPair(from0: Int, until0: Int,
                        fn:    (Int, Int) => Unit)
  : Unit = {
    var off0 = from0
    while (off0 < until0) {
      fn(off0 - from0, off0)
      off0 += 1
    }
  }

  @inline
  final def foreachPair(from0: Int, until0: Int, stride0: Int,
                        fn:    (Int, Int) => Unit)
  : Unit = {
    var off0 = from0
    var i    = 0
    val n    = noStepsFor(from0, until0, stride0)
    while (i < n) {
      fn(i, off0)
      off0 += stride0
      i    += 1
    }
  }

  @inline
  final def foreachPair(range0: Range,
                        fn:     (Int, Int) => Unit)
  : Unit = {
    val stride0 = range0.step
    var off0    = range0.start
    val length0 = range0.length

    var i = 0
    while (i < length0) {
      fn(i, off0)
      off0 += stride0
      i    += 1
    }
  }

  @inline
  final def foreachPair(range0: Range,
                        range1: Range,
                        fn:     (Int, Int, Int) => Unit)
  : Unit = {
    val stride1 = range1.step
    var off1    = range1.start
    val length1 = range1.length
    val stride0 = range0.step
    var off0    = range0.start
    val length0 = range0.length

    require(length0 == length1)

    var i = 0
    while (i < length0) {
      fn(i, off0, off1)
      off1 += stride1
      off0 += stride0
      i    += 1
    }
  }

  @inline
  final def foreachPair(range0: Range,
                        range1: Range,
                        range2: Range,
                        fn:     (Int, Int, Int, Int) => Unit)
  : Unit = {
    val stride2 = range2.step
    var off2    = range2.start
    val length2 = range2.length
    val stride1 = range1.step
    var off1    = range1.start
    val length1 = range1.length
    val stride0 = range0.step
    var off0    = range0.start
    val length0 = range0.length

    require(
      length0 == length1 &&
      length0 == length2
    )

    var i = 0
    while (i < length0) {
      fn(i, off0, off1, off2)
      off2 += stride2
      off1 += stride1
      off0 += stride0
      i    += 1
    }
  }

  @inline
  final def foreachPairParallel(from0: Int, until0: Int,
                                fn:    (Int, Int) => Unit)
  : Unit = {
    val tasks = mapPairs(
      from0, until0
    )((i, off0) => Future(fn(i, off0)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachPairParallel(from0: Int, until0: Int, stride0: Int,
                                fn:    (Int, Int) => Unit)
  : Unit = {
    val tasks = mapPairs(
      from0, until0, stride0
    )((i, off0) => Future(fn(i, off0)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachPairParallel(range0: Range,
                                fn:     (Int, Int) => Unit)
  : Unit = {
    val tasks = mapPairs(
      range0
    )((i, off0) => Future(fn(i, off0)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachPairParallel(range0: Range,
                                range1: Range,
                                fn:     (Int, Int, Int) => Unit)
  : Unit = {
    val tasks = mapPairs(
      range0,
      range1
    )((i, off0, off1) => Future(fn(i, off0, off1)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def foreachPairParallel(range0: Range,
                                range1: Range,
                                range2: Range,
                                fn:     (Int, Int, Int, Int) => Unit)
  : Unit = {
    val tasks = mapPairs(
      range0,
      range1,
      range2
    )((i, off0, off1, off2) => Future(fn(i, off0, off1, off2)))
    ArrayEx.finishAll(tasks)
  }

  @inline
  final def map[T](from0: Int, until0: Int)
                  (fn: Int => T)
                  (implicit tagT: ClassTag[T])
  : Array[T] = {
    ArrayEx.tabulate(
      noStepsFor(from0, until0)
    )(i => fn(from0 + i))
  }

  @inline
  final def map[T](from0: Int, until0: Int, stride0: Int)
                  (fn: Int => T)
                  (implicit tagT: ClassTag[T])
  : Array[T] = {
    var off0 = from0
    ArrayEx.fill(
      noStepsFor(from0, until0, stride0)
    )({
      val tmp0 = fn(off0)
      off0 += stride0
      tmp0
    })
  }

  @inline
  final def map[T](range0: Range)
                  (fn: Int => T)
                  (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](range0.length)
    foreachPair(
      range0,
      (i, off0) => result(i) = fn(off0)
    )
    result
  }

  @inline
  final def map[T](range0: Range,
                   range1: Range)
                  (fn: (Int, Int) => T)
                  (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](range0.length)
    foreachPair(
      range0,
      range1,
      (i, off0, off1) => result(i) = fn(off0, off1)
    )
    result
  }

  @inline
  final def map[T](range0: Range,
                   range1: Range,
                   range2: Range)
                  (fn: (Int, Int, Int) => T)
                  (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](range0.length)
    foreachPair(
      range0,
      range1,
      range2,
      (i, off0, off1, off2) => result(i) = fn(off0, off1, off2)
    )
    result
  }

  @inline
  final def mapParallel[T](from0: Int, until0: Int)
                          (fn: Int => T)
                          (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = map(
      from0, until0
    )(off0 => Future(fn(off0)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapParallel[T](from0: Int, until0: Int, stride0: Int)
                          (fn: Int => T)
                          (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = map(
      from0, until0, stride0
    )(off0 => Future(fn(off0)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapParallel[T](range0: Range)
                          (fn: Int => T)
                          (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = map(
      range0
    )(off0 => Future(fn(off0)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapParallel[T](range0: Range,
                           range1: Range)
                          (fn: (Int, Int) => T)
                          (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = map(
      range0,
      range1
    )((off0, off1) => Future(fn(off0, off1)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapParallel[T](range0: Range,
                           range1: Range,
                           range2: Range)
                          (fn: (Int, Int, Int) => T)
                          (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = map(
      range0,
      range1,
      range2
    )((off0, off1, off2) => Future(fn(off0, off1, off2)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapPairs[T](from0: Int, until0: Int)
                       (fn: (Int, Int) => T)
                       (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](noStepsFor(from0, until0))
    foreachPair(
      from0, until0,
      (i, off0) => result(i) = fn(i, off0)
    )
    result
  }

  @inline
  final def mapPairs[T](from0: Int, until0: Int, stride0: Int)
                       (fn: (Int, Int) => T)
                       (implicit tagT: ClassTag[T])
  : Array[T] = {
    var off0 = from0
    ArrayEx.tabulate(
      noStepsFor(from0, until0, stride0)
    )(
      i => {
        val tmp0 = fn(i, off0)
        off0 += stride0
        tmp0
      }
    )
  }

  @inline
  final def mapPairs[T](range0: Range)
                       (fn: (Int, Int) => T)
                       (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](range0.length)
    foreachPair(
      range0,
      (i, off0) => result(i) = fn(i, off0)
    )
    result
  }

  @inline
  final def mapPairs[T](range0: Range,
                        range1: Range)
                       (fn: (Int, Int, Int) => T)
                       (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](range0.length)
    foreachPair(
      range0,
      range1,
      (i, off0, off1) => result(i) = fn(i, off0, off1)
    )
    result
  }

  @inline
  final def mapPairs[T](range0: Range,
                        range1: Range,
                        range2: Range)
                       (fn: (Int, Int, Int, Int) => T)
                       (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](range0.length)
    foreachPair(
      range0,
      range1,
      range2,
      (i, off0, off1, off2) => result(i) = fn(i, off0, off1, off2)
    )
    result
  }

  @inline
  final def mapPairsParallel[T](from0: Int, until0: Int)
                               (fn: (Int, Int) => T)
                               (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = mapPairs(
      from0, until0
    )((i, off0) => Future(fn(i, off0)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapPairsParallel[T](from0: Int, until0: Int, stride0: Int)
                               (fn: (Int, Int) => T)
                               (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = mapPairs(
      from0, until0, stride0
    )((i, off0) => Future(fn(i, off0)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapPairsParallel[T](range0: Range)
                               (fn: (Int, Int) => T)
                               (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = mapPairs(
      range0
    )((i, off0) => Future(fn(i, off0)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapPairsParallel[T](range0: Range,
                                range1: Range)
                               (fn: (Int, Int, Int) => T)
                               (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = mapPairs(
      range0,
      range1
    )((i, off0, off1) => Future(fn(i, off0, off1)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def mapPairsParallel[T](range0: Range,
                                range1: Range,
                                range2: Range)
                               (fn: (Int, Int, Int, Int) => T)
                               (implicit tagT: ClassTag[T])
  : Array[T] = {
    val tasks = mapPairs(
      range0,
      range1,
      range2
    )((i, off0, off1, off2) => Future(fn(i, off0, off1, off2)))
    ArrayEx.getAll(tasks)
  }

  @inline
  final def noStepsFor(from: Int, until: Int)
  : Int = Math.max(until - from, 0)

  @inline
  final def noStepsFor(from: Int, until: Int, stride: Int)
  : Int = Math.max((until - from) / stride, 0)

}
