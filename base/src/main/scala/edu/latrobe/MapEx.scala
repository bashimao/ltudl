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

import scala.collection._
import scala.reflect.ClassTag

object MapEx {

  @inline
  final def foldLeft[T, U, V](value0: T,
                              map1:   Map[U, V])
                             (fn: (T, U, V) => T)
  : T = {
    var result = value0
    foreach(map1)(
      (k, v) => result = fn(result, k, v)
    )
    result
  }

  @inline
  final def foldLeft[T, U, V, W](value0: T,
                                 map1:   SortedMap[U, V],
                                 map2:   SortedMap[U, W])
                                (fn: (T, U, V, W) => T)
  : T = {
    var result = value0
    foreach(map1, map2)(
      (k, v1, v2) => result = fn(result, k, v1, v2)
    )
    result
  }

  @inline
  final def foldLeftEx[T, U, V, W](value0: T,
                                   map1:   Map[U, V],
                                   map2:   Map[U, W])
                                  (fn0: (T, U, V, W) => T, fn1: (T, U, V) => T, fn2: (T, U, W) => T)
  : T = {
    var result = value0
    foreachEx(map1, map2)(
      (k, v1, v2) => result = fn0(result, k, v1, v2),
      (k, v1)     => result = fn1(result, k, v1),
      (k, v2)     => result = fn2(result, k, v2)
    )
    result
  }

  @inline
  final def foldLeftValues[T, U, V](value0: T,
                                    map1:   Map[U, V])
                                   (fn: (T, V) => T)
  : T = {
    var result = value0
    foreachValue(map1)(
      v1 => result = fn(result, v1)
    )
    result
  }

  @inline
  final def foldLeftValues[T, U, V, W](value0: T,
                                       map1:   SortedMap[U, V],
                                       map2:   SortedMap[U, W])
                                      (fn: (T, V, W) => T)
  : T = {
    var result = value0
    foreachValue(map1, map2)(
      (v1, v2) => result = fn(result, v1, v2)
    )
    result
  }

  @inline
  final def foldLeftValuesEx[T, U, V, W](value0: T,
                                         map1:   Map[U, V],
                                         map2:   Map[U, W])
                                        (fn0: (T, V, W) => T, fn1: (T, V) => T, fn2: (T, W) => T)
  : T = {
    var result = value0
    foreachValueEx(map1, map2)(
      (v1, v2) => result = fn0(result, v1, v2),
      v1       => result = fn1(result, v1),
      v2       => result = fn2(result, v2)
    )
    result
  }

  @inline
  final def filter[T, U](map0: Map[T, U])
                        (fn: (T, U) => Boolean)
  : Map[T, U] = map0.filter(kv => fn(kv._1, kv._2))

  @inline
  final def filter[T, U](map0: SortedMap[T, U])
                        (fn: (T, U) => Boolean)
  : SortedMap[T, U] = map0.filter(kv => fn(kv._1, kv._2))


  @inline
  final def filterByKey[T, U](map0: Map[T, U])
                             (fn: T => Boolean)
  : Map[T, U] = map0.filter(kv => fn(kv._1))

  @inline
  final def filterByKey[T, U](map0: SortedMap[T, U])
                             (fn: T => Boolean)
  : SortedMap[T, U] = map0.filter(kv => fn(kv._1))

  @inline
  final def filterByValue[T, U](map0: Map[T, U])
                               (fn: U => Boolean)
  : Map[T, U] = map0.filter(kv => fn(kv._2))

  @inline
  final def filterByValue[T, U](map0: SortedMap[T, U])
                               (fn: U => Boolean)
  : SortedMap[T, U] = map0.filter(kv => fn(kv._2))

  @inline
  final def foreach[T, U](map0: Map[T, U])
                         (fn: (T, U) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    while (iter0.hasNext) {
      val kv0 = iter0.next()
      fn(kv0._1, kv0._2)
    }
  }

  @inline
  final def foreach[T, U, V](map0: SortedMap[T, U],
                             map1: SortedMap[T, V])
                            (fn: (T, U, V) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    while (iter0.hasNext) {
      val kv0 = iter0.next()
      val kv1 = iter1.next()
      assume(kv0._1 == kv1._1)
      fn(kv0._1, kv0._2, kv1._2)
    }
    assume(!iter1.hasNext)
  }

  @inline
  final def foreach[T, U, V, W](map0: SortedMap[T, U],
                                map1: SortedMap[T, V],
                                map2: SortedMap[T, W])
                               (fn: (T, U, V, W) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    val iter2 = map2.iterator
    while (iter0.hasNext) {
      val kv0 = iter0.next()
      val kv1 = iter1.next()
      val kv2 = iter2.next()
      assume(
        kv0._1 == kv1._1 &&
        kv0._1 == kv2._1
      )
      fn(kv0._1, kv0._2, kv1._2, kv2._2)
    }
    assume(!iter1.hasNext && !iter2.hasNext)
  }

  @inline
  final def foreachEx[T, U, V](map0: Map[T, U],
                               map1: Map[T, V])
                              (fn0: (T, U, V) => Unit, fn1: (T, U) => Unit, fn2: (T, V) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val v1 = map1.get(k0)
      if (v1.isDefined) {
        fn0(k0, v0, v1.get)
      }
      else {
        fn1(k0, v0)
      }
    }
    val iter1 = map1.iterator
    while (iter1.hasNext) {
      val (k1, v1) = iter1.next()
      if (!map0.contains(k1)) {
        fn2(k1, v1)
      }
    }
  }

  @inline
  final def foreachKey[T, U](map0: Map[T, U])
                            (fn: T => Unit)
  : Unit = {
    val iter0 = map0.iterator
    while (iter0.hasNext) {
      fn(iter0.next()._1)
    }
  }

  @inline
  final def foreachPair[T, U](map0: Map[T, U])
                             (fn: (Int, T, U) => Unit)
  : Unit = {
    var i     = 0
    val iter0 = map0.iterator
    while (iter0.hasNext) {
      val kv0 = iter0.next()
      fn(i, kv0._1, kv0._2)
      i += 1
    }
  }

  @inline
  final def foreachValue[T, U](map0: Map[T, U])
                              (fn: U => Unit)
  : Unit = {
    val iter0 = map0.valuesIterator
    while (iter0.hasNext) {
      fn(iter0.next())
    }
  }

  @inline
  final def foreachValue[T, U, V](map0: SortedMap[T, U],
                                  map1: SortedMap[T, V])
                                 (fn: (U, V) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val (k1, v1) = iter1.next()
      assume(k0 == k1)
      fn(v0, v1)
    }
    assume(!iter1.hasNext)
  }

  @inline
  final def foreachValue[T, U, V, W](map0: SortedMap[T, U],
                                     map1: SortedMap[T, V],
                                     map2: SortedMap[T, W])
                                    (fn: (U, V, W) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    val iter2 = map2.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val (k1, v1) = iter1.next()
      val (k2, v2) = iter2.next()
      assume(
        k0 == k1 &&
        k0 == k2
      )
      fn(v0, v1, v2)
    }
    assume(
      !iter1.hasNext &&
      !iter2.hasNext
    )
  }

  @inline
  final def foreachValue[T, U, V, W, X](map0: SortedMap[T, U],
                                        map1: SortedMap[T, V],
                                        map2: SortedMap[T, W],
                                        map3: SortedMap[T, X])
                                       (fn: (U, V, W, X) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    val iter2 = map2.iterator
    val iter3 = map3.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val (k1, v1) = iter1.next()
      val (k2, v2) = iter2.next()
      val (k3, v3) = iter3.next()
      assume(
        k0 == k1 &&
        k0 == k2 &&
        k0 == k3
      )
      fn(v0, v1, v2, v3)
    }
    assume(
      !iter1.hasNext &&
      !iter2.hasNext &&
      !iter3.hasNext
    )
  }

  @inline
  final def foreachValue[T, U, V, W, X, Y](map0: SortedMap[T, U],
                                           map1: SortedMap[T, V],
                                           map2: SortedMap[T, W],
                                           map3: SortedMap[T, X],
                                           map4: SortedMap[T, Y])
                                          (fn: (U, V, W, X, Y) => Unit)
  : Unit = {
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    val iter2 = map2.iterator
    val iter3 = map3.iterator
    val iter4 = map4.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val (k1, v1) = iter1.next()
      val (k2, v2) = iter2.next()
      val (k3, v3) = iter3.next()
      val (k4, v4) = iter4.next()
      assume(
        k0 == k1 &&
        k0 == k2 &&
        k0 == k3 &&
        k0 == k4
      )
      fn(v0, v1, v2, v3, v4)
    }
    assume(
      !iter1.hasNext &&
      !iter2.hasNext &&
      !iter3.hasNext &&
      !iter4.hasNext
    )
  }

  @inline
  final def foreachValueEx[T, U, V](map0: Map[T, U],
                                    map1: Map[T, V])
                                   (fn0: (U, V) => Unit, fn1: U => Unit, fn2: V => Unit)
  : Unit = {
    val iter0 = map0.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val v1 = map1.get(k0)
      if (v1.isDefined) {
        fn0(v0, v1.get)
      }
      else {
        fn1(v0)
      }
    }
    val iter1 = map1.iterator
    while (iter1.hasNext) {
      val (k1, v1) = iter1.next()
      if (!map0.contains(k1)) {
        fn2(v1)
      }
    }
  }

  @inline
  final def map[T, U, V](map0: Map[T, U])
                        (fn: (T, U) => V)
  : Map[T, V] = map0.map(kv => Tuple2(kv._1, fn(kv._1, kv._2)))

  @inline
  final def map[T, U, V](map0: SortedMap[T, U])
                        (fn: (T, U) => V)
                        (implicit orderingT: Ordering[T])
  : SortedMap[T, V] = map0.map(kv => Tuple2(kv._1, fn(kv._1, kv._2)))

  @inline
  final def mapValuesEx[T, U, V, W](map0: Map[T, U],
                                    map1: Map[T, V])
                                   (fn0: (U, V) => W, fn1: U => W, fn2: V => W)
                                   (implicit orderingT: Ordering[T])
  : SortedMap[T, W] = {
    val builder = SortedMap.newBuilder[T, W]
    foreachEx(
      map0,
      map1
    )(
      builder += _ -> fn0(_, _),
      builder += _ -> fn1(_),
      builder += _ -> fn2(_)
    )
    builder.result()
  }

  /**
    * Shorthand that does not create temporary map.
    */
  @inline
  final def mapReduceLeftValues[T, U, V](map1: Map[T, U])
                                        (mapFn: U => V)
                                        (reduceFn: (V, V) => V)
  : V = {
    val iter1  = map1.valuesIterator
    var result = mapFn(iter1.next())
    while (iter1.hasNext) {
      result = reduceFn(result, mapFn(iter1.next()))
    }
    result
  }

  @inline
  final def mapValues[T, U, V](map0: Map[T, U])
                              (fn: U => V)
  : Map[T, V] = map0.map(
    kv => Tuple2(kv._1, fn(kv._2))
  )

  @inline
  final def mapValues[T, U, V](map0: SortedMap[T, U])
                              (fn: U => V)
                              (implicit orderingT: Ordering[T])
  : SortedMap[T, V] = map0.map(
    kv => Tuple2(kv._1, fn(kv._2))
  )

  @inline
  final def reduceLeftValues[T, U](map1: Map[T, U])
                                  (fn: (U, U) => U)
  : U = {
    val iter1  = map1.valuesIterator
    var result = iter1.next()
    while (iter1.hasNext) {
      result = fn(result, iter1.next())
    }
    result
  }

  @inline
  final def toArray[T, U](length: Int, map0: Map[Int, T])
                         (fn: (Int, T) => U)
                         (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](length)
    foreach(map0)(
      (k, v) => result(k) = fn(k, v)
    )
    result
  }

  @inline
  final def zip[T, U, V, W](map0: SortedMap[T, U],
                            map1: SortedMap[T, V])
                           (fn: (T, U, V) => W)
                           (implicit orderingT: Ordering[T])
  : SortedMap[T, W] = {
    val builder = SortedMap.newBuilder[T, W]
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val (k1, v1) = iter1.next()
      assume(k0 == k1)
      builder += Tuple2(k0, fn(k0, v0, v1))
    }
    assume(!iter1.hasNext)
    builder.result()
  }

  @inline
  final def zip[T, U, V, W, X](map0: SortedMap[T, U],
                               map1: SortedMap[T, V],
                               map2: SortedMap[T, W])
                              (fn: (T, U, V, W) => X)
                              (implicit orderingT: Ordering[T])
  : SortedMap[T, X] = {
    val builder = SortedMap.newBuilder[T, X]
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    val iter2 = map2.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val (k1, v1) = iter1.next()
      val (k2, v2) = iter2.next()
      assume(k0 == k1 && k0 == k2)
      builder += Tuple2(k0, fn(k0, v0, v1, v2))
    }
    assume(!iter1.hasNext)
    builder.result()
  }

  @inline
  final def zipEx[T, U, V, W](map0: Map[T, U],
                              map1: Map[T, V])
                             (fn0: (T, U, V) => W, fn1: (T, U) => W, fn2: (T, V) => W)
                             (implicit orderingT: Ordering[T])
  : SortedMap[T, W] = {
    val builder = SortedMap.newBuilder[T, W]
    foreach(map0)(
      (k0, v0) => {
        val v1 = map1.get(k0)
        if (v1.isDefined) {
          builder += Tuple2(k0, fn0(k0, v0, v1.get))
        }
        else {
          builder += Tuple2(k0, fn1(k0, v0))
        }
      }
    )
    foreach(map1)(
      (k1, v1) => {
        if (!map0.contains(k1)) {
          builder += Tuple2(k1, fn2(k1, v1))
        }
      }
    )
    builder.result()
  }

  @inline
  final def zipValues[T, U, V, W](map0: SortedMap[T, U],
                                  map1: SortedMap[T, V])
                                 (fn: (U, V) => W)
                                 (implicit orderingT: Ordering[T])
  : SortedMap[T, W] = {
    val builder = SortedMap.newBuilder[T, W]
    val iter0 = map0.iterator
    val iter1 = map1.iterator
    while (iter0.hasNext) {
      val (k0, v0) = iter0.next()
      val (k1, v1) = iter1.next()
      assume(k0 == k1)
      builder += Tuple2(k0, fn(v0, v1))
    }
    assume(!iter1.hasNext)
    builder.result()
  }

  @inline
  final def zipValuesEx[T, U, V, W](map0: Map[T, U],
                                    map1: Map[T, V])
                                   (fn0: (U, V) => W, fn1: U => W, fn2: V => W)
  : Map[T, W] = {
    val builder = Map.newBuilder[T, W]
    foreach(map0)(
      (k0, v0) => {
        val v1 = map1.get(k0)
        if (v1.isDefined) {
          builder += Tuple2(k0, fn0(v0, v1.get))
        }
        else {
          builder += Tuple2(k0, fn1(v0))
        }
      }
    )
    foreach(map1)(
      (k1, v1) => {
        if (!map0.contains(k1)) {
          builder += Tuple2(k1, fn2(v1))
        }
      }
    )
    builder.result()
  }

  @inline
  final def zipValuesEx[T, U, V, W](map0: SortedMap[T, U],
                                    map1: SortedMap[T, V])
                                   (fn0: (U, V) => W, fn1: U => W, fn2: V => W)
                                   (implicit orderingT: Ordering[T])
  : SortedMap[T, W] = {
    val builder = SortedMap.newBuilder[T, W]
    foreach(map0)(
      (k0, v0) => {
        val v1 = map1.get(k0)
        if (v1.isDefined) {
          builder += Tuple2(k0, fn0(v0, v1.get))
        }
        else {
          builder += Tuple2(k0, fn1(v0))
        }
      }
    )
    foreach(map1)(
      (k1, v1) => {
        if (!map0.contains(k1)) {
          builder += Tuple2(k1, fn2(v1))
        }
      }
    )
    builder.result()
  }

}
