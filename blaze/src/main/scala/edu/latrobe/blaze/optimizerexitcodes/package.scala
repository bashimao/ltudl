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

import breeze.optimize.FirstOrderMinimizer._

/**
  * Created by tl on 4/13/16.
  */
package object optimizerexitcodes {

  final implicit class ConvergenceReasonFunctions(cr: ConvergenceReason) {

    def toOptimizationResult(iterationNo: Long,
                             noSamples:   Long)
    : OptimizationResult = cr match {

      case MaxIterations =>
        OptimizationResult.derive(
          NoIterationsLimit(),
          iterationNo,
          noSamples
        )

      case FunctionValuesConverged =>
        OptimizationResult.derive(
          ThirdParty.convergence(cr.reason),
          iterationNo,
          noSamples
        )

      case GradientConverged =>
        OptimizationResult.derive(
          ThirdParty.convergence(cr.reason),
          iterationNo,
          noSamples
        )

      case SearchFailed =>
        OptimizationResult.derive(
          LineSearchFailed(),
          iterationNo,
          noSamples
        )

      case ObjectiveNotImproving =>
        OptimizationResult.derive(
          ThirdParty.failure(cr.reason),
          iterationNo,
          noSamples
        )

      // TODO: Update once we have newer breeze support.
      /*
      case MonitorFunctionNotImproving =>
        OptimizationResult.derive(
          ThirdParty.failure(cr.reason),
          iterationNo,
          noSamples
        )

      case ProjectedStepConverged =>
        OptimizationResult.derive(
          ThirdParty.convergence(cr.reason),
          iterationNo,
          noSamples
        )
      */

      case _ =>
        throw new MatchError(cr)
    }
  }

}
