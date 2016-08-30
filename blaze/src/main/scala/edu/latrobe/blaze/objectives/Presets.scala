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

package edu.latrobe.blaze.objectives

import edu.latrobe.blaze.objectives.visual._
import edu.latrobe.blaze.validators._

object Presets {

  final def printValueCSV(format: String = "%.4g")
  : ComplexObjectiveBuilder = {
    ComplexObjectiveBuilder(
      SelectTimeInSecondsBuilder() += PrintValueBuilder.derive("%.3f"),
      PrintStringBuilder(", "),
      PrintRunNoBuilder(),
      PrintStringBuilder(", "),
      PrintIterationNoBuilder(),
      PrintStringBuilder(", "),
      PrintValueBuilder.derive(format),
      PrintStringBuilder.lineSeparator
    )
  }

  final def visualizeCommonParameters(xAxisType: XAxisType = XAxisType.IterationNo)
  : VisualizeCurvesBuilder = {
    val result = VisualizeCurvesBuilder()
    result.xAxisType = xAxisType
    result.yAxis0Curves ++= Seq(
      ParameterValueCurveBuilder(
        "learningRate", "learning rate"
      ).setWindowType(CurveWindowType.Compressing(256)),
      ParameterValueCurveBuilder(
        "scalaCoefficient", "lambda"
      ).setWindowType(CurveWindowType.Compressing(256))
    )
    result.yAxis1Curves ++= Seq(
      ParameterValueCurveBuilder(
        "decayRate", "decay rate"
      ).setWindowType(CurveWindowType.Compressing(256)),
      ParameterValueCurveBuilder(
        "dampeningFactor", "dampening factor"
      ).setWindowType(CurveWindowType.Compressing(256))
    )
    result
  }

  final def visualizePerformance(xAxisType: XAxisType = XAxisType.IterationNo)
  : VisualizeCurvesBuilder = {
    val result = VisualizeCurvesBuilder()
    result.xAxisType = xAxisType
    result.yAxis0Curves ++= Seq(
      ValueCurveBuilder().setWindowType(CurveWindowType.Compressing(512)),
      ValueCurveBuilder().transformLabel(
        x => s"smooth($x)"
      ).setWindowType(CurveWindowType.Compressing(48))
    )
    result.yAxis1Curves ++= Seq(
      ValidationCurveBuilder(Top1LabelValidatorBuilder()).setLabel(
        "Top1 acc. [%]"
      ).setWindowType(CurveWindowType.Compressing(512)),
      ValidationCurveBuilder(Top1LabelValidatorBuilder()).setLabel(
        "smooth(Top1 acc. [%])"
      ).setWindowType(CurveWindowType.Compressing(48)),
      ValidationCurveBuilder(TopKLabelsValidatorBuilder(5)).setLabel(
        "Top5 acc. [%]"
      ).setWindowType(CurveWindowType.Compressing(512)),
      ValidationCurveBuilder(TopKLabelsValidatorBuilder(5)).setLabel(
        "smooth(Top5 acc. [%])"
      ).setWindowType(CurveWindowType.Compressing(48))
    )
    result
  }

  final def visualizePerformanceLight(xAxisType: XAxisType = XAxisType.IterationNo)
  : VisualizeCurvesBuilder = {
    val result = VisualizeCurvesBuilder()
    result.xAxisType = xAxisType
    result.yAxis0Curves ++= Seq(
      ValueCurveBuilder().setWindowType(CurveWindowType.Compressing(512)),
      ValueCurveBuilder().transformLabel(
        x => s"smooth($x)"
      ).setWindowType(CurveWindowType.Compressing(48))
    )
    result
  }

  final def visualizeRuntimeStatistics(xAxisType: XAxisType = XAxisType.TimeInSeconds)
  : VisualizeCurvesBuilder = {
    val result = VisualizeCurvesBuilder()
    result.xAxisType = xAxisType
    result.yAxis0Curves ++= Seq(
      IterationNoCurveBuilder().setWindowType(CurveWindowType.Compressing(128)),
      RunNoCurveCurveBuilder().setWindowType(CurveWindowType.Compressing(128))
    )
    result.yAxis1Curves ++= Seq(
      SamplesPerSecondCurveBuilder().setWindowType(CurveWindowType.Compressing(256)),
      SamplesPerSecondCurveBuilder().transformLabel(
        x => s"smooth($x)"
      ).setWindowType(CurveWindowType.Compressing(32))
    )
    result
  }

}
