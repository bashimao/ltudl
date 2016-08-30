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

import java.util.concurrent.Callable

import scala.collection._
import scala.collection.convert.Wrappers._
import scala.collection.parallel.ForkJoinTasks
import scala.reflect._

object SeqEx {

  val pool = ForkJoinTasks.defaultForkJoinPool

  final def concat[T](value0: T, values: TraversableOnce[T])
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val builder = Array.newBuilder[T]
    builder += value0
    builder ++= values
    builder.result()
  }

  /*
  final def concatFlat[T](value0: Array[T], valueN: Array[T]*)
  (implicit tagT: ClassTag[T])
  : Array[T] = {
    val builder = Array.newBuilder[T]
    builder ++= value0
    valueN.foreach(
      builder ++= _
    )
    builder.result()
  }

  final def concatFlat[T](values: TraversableOnce[Array[T]])
                         (implicit tagT: ClassTag[T])
  : Array[T] = {
    val builder = Array.newBuilder[T]
    values.foreach(
      builder ++= _
    )
    builder.result()
  }
  */

  @inline
  final def execute[T](seq0: Seq[Callable[T]])
  : Unit = pool.invokeAll(SeqWrapper(seq0))

  @inline
  final def foldLeft[T, U, V](value0: T,
                              seq0: Iterable[U],
                              seq1: Iterable[V])
                             (fn: (T, U, V) => T)
  : T = {
    var result = value0
    foreach(seq0, seq1)(
      (v0, v1) => result = fn(result, v0, v1)
    )
    result
  }

  @inline
  final def foldLeft[T, U, V, W](value0: T,
                                 seq0: Iterable[U],
                                 seq1: Iterable[V],
                                 seq2: Iterable[W])
                                (fn: (T, U, V, W) => T)
  : T = {
    var result = value0
    foreach(seq0, seq1, seq2)(
      (v0, v1, v2) => result = fn(result, v0, v1, v2)
    )
    result
  }

  @inline
  final def foldLeftPairs[T, U](value0: T,
                                seq1: TraversableOnce[U])
                               (fn: (T, Int, U) => T)
  : T = {
    var result = value0
    foreachPair(seq1)(
      (i, v1) => result = fn(result, i, v1)
    )
    result
  }

  @inline
  final def foreach[T, U](seq0: Iterable[T],
                          seq1: Iterable[U])
                         (fn: (T, U) => Unit)
  : Unit = {
    val iter0 = seq0.iterator
    val iter1 = seq1.iterator
    while (iter0.hasNext) {
      fn(
        iter0.next(),
        iter1.next()
      )
    }
    assume(!iter1.hasNext)
  }

  @inline
  final def foreach[T, U, V](seq0: Iterable[T],
                             seq1: Iterable[U],
                             seq2: Iterable[V])
                            (fn: (T, U, V) => Unit)
  : Unit = {
    val iter0 = seq0.iterator
    val iter1 = seq1.iterator
    val iter2 = seq2.iterator
    while (iter0.hasNext) {
      fn(
        iter0.next(),
        iter1.next(),
        iter2.next()
      )
    }
    assume(!iter1.hasNext && !iter2.hasNext)
  }

  @inline
  final def foreachPair[T](seq0: TraversableOnce[T])
                          (fn: (Int, T) => Unit)
  : Unit = {
    var i = 0
    seq0.foreach(v => {
      fn(i, v)
      i += 1
    })
  }

  @inline
  final def foreachPair[T, U](seq0: Iterable[T],
                              seq1: Iterable[U])
                             (fn: (Int, T, U) => Unit)
  : Unit = {
    var i     = 0
    val iter0 = seq0.iterator
    val iter1 = seq1.iterator
    while (iter0.hasNext) {
      fn(
        i,
        iter0.next(),
        iter1.next()
      )
      i += 1
    }
    assume(!iter1.hasNext)
  }

  @inline
  final def foreachPair[T, U, V](seq0: Iterable[T],
                                 seq1: Iterable[U],
                                 seq2: Iterable[V])
                                (fn: (Int, T, U, V) => Unit)
  : Unit = {
    var i     = 0
    val iter0 = seq0.iterator
    val iter1 = seq1.iterator
    val iter2 = seq2.iterator
    while (iter0.hasNext) {
      fn(
        i,
        iter0.next(),
        iter1.next(),
        iter2.next()
      )
      i += 1
    }
    assume(!iter1.hasNext && !iter2.hasNext)
  }

  @inline
  final def mapPairs[T, U](seq0: TraversableOnce[T])
                          (fn: (Int, T) => U)
                          (implicit tagT: ClassTag[U])
  : Array[U] = {
    val builder = Array.newBuilder[U]
    foreachPair(
      seq0
    )(builder += fn(_, _))
    builder.result()
  }

  @inline
  final def slice[T](seq0: Seq[T], n: Range)
                    (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](n.length)
    var i      = 0
    var j      = n.start
    while (i < result.length) {
      result(i) = seq0(j)
      j += n.step
      i += 1
    }
    result
  }

  @inline
  final def zip[T, U, V](seq0: Iterable[T],
                         seq1: Iterable[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = Array.newBuilder[V]
    foreach(
      seq0,
      seq1
    )((v0, v1) => result += fn(v0, v1))
    result.result()
  }

  @inline
  final def zip[T, U, V, W](seq0: Iterable[T],
                            seq1: Iterable[U],
                            seq2: Iterable[V])
                           (fn: (T, U, V) => W)
                           (implicit tagW: ClassTag[W])
  : Array[W] = {
    val result = Array.newBuilder[W]
    foreach(
      seq0,
      seq1,
      seq2
    )((v0, v1, v2) => result += fn(v0, v1, v2))
    result.result()
  }

  @inline
  final def zipPairs[T, U, V](seq0: Iterable[T],
                              seq1: Iterable[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = Array.newBuilder[V]
    foreachPair(
      seq0,
      seq1
    )((i, v0, v1) => result += fn(i, v0, v1))
    result.result()
  }

  @inline
  final def zipPairs[T, U, V, W](seq0: Iterable[T],
                                 seq1: Iterable[U],
                                 seq2: Iterable[V])
                                (fn: (Int, T, U, V) => W)
                                (implicit tagW: ClassTag[W])
  : Array[W] = {
    val result = Array.newBuilder[W]
    foreachPair(
      seq0,
      seq1,
      seq2
    )((i, v0, v1, v2) => result += fn(i, v0, v1, v2))
    result.result()
  }

}
