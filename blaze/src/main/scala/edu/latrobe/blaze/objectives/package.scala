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

package edu.latrobe.blaze

import edu.latrobe.io._
import edu.latrobe.blaze.sinks._
import scala.collection._
import scala.language.implicitConversions

/**
  * Here we define major parts of the DSL for objectives.
  *
  * Note that this is the operator precedence in Scala!
  * http://scala-lang.org/files/archive/spec/2.11/06-expressions.html#infix-operations
  *
  * (all letters)
  * |
  * ^
  * &
  * = !
  * < >
  * :
  * + -
  * * / %
  * (all other special characters)
  *
  */
package object objectives {

  object Implicits {

    implicit def sink2OutputRedirection(sink: SinkBuilder)
    : OutputRedirectionBuilder = OutputRedirectionBuilder(sink)

  }

  final implicit class ObjectiveBuilderFunctions(obj: ObjectiveBuilder) {

    def benchmark()
    : objectives.BenchmarkObjectiveBuilder = BenchmarkObjectiveBuilder(obj)


    // ---------------------------------------------------------------------------
    //    DSL
    // ---------------------------------------------------------------------------
    /**
      * Priority 1:
      *
      * ! can wrap around trigger objectives and inverts their result.
      */
    def unary_!()
    : InvertTriggerBuilder = InvertTriggerBuilder(obj)

    /**
      * Priority 2:
      *
      * && will bundle objectives together to form a complex objective that
      * is only met if all sub-objectives evaluate true. The objectives are
      * evaluated first to last. The first objectives that evaluates to None
      * will break the execution.
      */
    def &&(other: ObjectiveBuilder)
    : ComplexObjectiveBuilder = obj match {
      case obj: ComplexObjectiveBuilder =>
        obj.children += other
        obj
      case _ =>
        ComplexObjectiveBuilder(obj, other)
    }

    /**
      * Priority 3:
      *
      * || will group objectives together. They will be evaluated front to back.
      * The first objective that evaluates to anything except None will break
      * execution.
      */
    def ||(other: ObjectiveBuilder)
    : MultiObjectiveBuilder = obj match {
      case obj: MultiObjectiveBuilder =>
        obj.children += other
        obj
      case _ =>
        MultiObjectiveBuilder(obj, other)
    }

    /**
      * Priority 4:
      *
      * >& will redirect the output from the left hand objective into a
      * a sink.
      */
    def >>(other: OutputRedirectionBuilder)
    : OutputRedirectionBuilder = {
      other.children += obj
      other
    }

    /**
      * Priority 4:
      *
      * >& will redirect the output from the left hand objective into a
      * a sink.
      */
    def >>(other: SinkBuilder)
    : OutputRedirectionBuilder = OutputRedirectionBuilder(other, obj)

  }

}
