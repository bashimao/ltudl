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

import spire.macros.compat._

/**
  * I already fixed this. Wait until spire project takes over my pull request
  * and then replace this: https://github.com/non/spire/issues/556
  */
final class FixedInlineUtil[C <: Context with Singleton](val c: C) {
  import c.universe._
  // This is Scala reflection source compatibility hack between Scala 2.10 and 2.11

  def inlineAndReset[T](tree: Tree): c.Expr[T] = {
    val inlined = inlineApplyRecursive(tree)
    c.Expr[T](resetLocalAttrs(c)(inlined))
  }

  def inlineApplyRecursive(tree: Tree): Tree = {
    val ApplyName = termName(c)("apply")

    class InlineSymbol(name: TermName, symbol: Symbol, value: Tree) extends Transformer {
      override def transform(tree: Tree): Tree = tree match {
        case tree: Ident if tree.symbol == symbol =>
          if (tree.name == name) {
            value
          }
          else {
            super.transform(tree)
          }

        case tt: TypeTree if tt.original != null =>
          //super.transform(TypeTree().setOriginal(transform(tt.original)))
          super.transform(setOrig(c)(TypeTree(), transform(tt.original)))
        case _ =>
          super.transform(tree)
      }
    }

    object InlineApply extends Transformer {
      def inlineSymbol(name: TermName, symbol: Symbol, body: Tree, arg: Tree): Tree =
        new InlineSymbol(name, symbol, arg).transform(body)

      override def transform(tree: Tree): Tree = tree match {
        case Apply(Select(Function(params, body), ApplyName), args) =>
          params.zip(args).foldLeft(body) { case (b, (param, arg)) =>
            inlineSymbol(param.name, param.symbol, b, arg)
          }

        case Apply(Function(params, body), args) =>
          params.zip(args).foldLeft(body) { case (b, (param, arg)) =>
            inlineSymbol(param.name, param.symbol, b, arg)
          }

        case _ =>
          super.transform(tree)
      }
    }

    InlineApply.transform(tree)
  }
}
