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

package edu.latrobe.blaze.modules

import breeze.linalg.DenseMatrix
import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * Transforms input through multiplying it with a matrix. This is quite useful
  * in pre-processing but might also have other applications.
  *
  * Can be used to extract luminance from a RGB input.
  */
// TODO: Make use of tensor functions to achieve this to enable CUDA support.
final class MatrixTransform(override val builder:        MatrixTransformBuilder,
                            override val inputHints:     BuildHints,
                            override val seed:           InstanceSeed,
                            override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[MatrixTransformBuilder]
    with NonTrainableLayer[MatrixTransformBuilder]
    with NonPenalizing {

  override val outputHints
  : BuildHints = builder.outputHintsFor(inputHints)

  private val matrix
  : DenseMatrix[Real] = builder.matrix

  private lazy val matrixInv
  : DenseMatrix[Real] = {
    val tmp = matrix.copy
    _LAPACK.inv(tmp)
    tmp
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val inpSize = input.layout.size
    require(inpSize.noChannels == matrix.cols)
    val outSize = inpSize.withNoChannels(matrix.rows)

    val inp = MatrixEx.reshape(input.valuesMatrix, matrix.cols)
    val out = matrix * inp

    (RealArrayTensor.derive(outSize, out), SizeContext(inpSize))
  }

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = context match {
    case SizeContext(inpSize) =>
      val out = output.valuesMatrix
      val inp = matrixInv * out
      RealArrayTensor.derive(inpSize, inp)
    case _ =>
      throw new MatchError(context)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveInputError(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor = context match {
    case SizeContext(inpSize) =>
      val oldErr = error.valuesMatrix
      val newErr = matrixInv * oldErr
      RealArrayTensor.derive(inpSize, newErr)
    case _ =>
      throw new MatchError(context)
  }


}

final class MatrixTransformBuilder
  extends LayerBuilder[MatrixTransformBuilder]
    with NonTrainableLayerBuilder[MatrixTransformBuilder] {

  override def repr
  : MatrixTransformBuilder = this

  private var _matrix
  : DenseMatrix[Real] = MatrixEx.eye(1, 1)

  def matrix
  : DenseMatrix[Real] = _matrix

  def matrix_=(value: DenseMatrix[Real])
  : Unit = {
    require(value != null)
    _matrix = value
  }

  def setMatrix(value: DenseMatrix[Real])
  : MatrixTransformBuilder = {
    matrix_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = s"${_matrix.rows} x ${_matrix.cols}" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _matrix.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MatrixTransformBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MatrixTransformBuilder =>
      _matrix == other._matrix
    case _ =>
      false
  })

  override protected def doCopy()
  : MatrixTransformBuilder = MatrixTransformBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: MatrixTransformBuilder =>
        other._matrix = _matrix
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  def outputLayoutFor(inputLayout: TensorLayout)
  : IndependentTensorLayout = inputLayout.derive(
    inputLayout.size.withNoChannels(_matrix.cols)
  )

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(JVM, outputLayoutFor(hints.layout))

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = new MatrixTransform(this, hints, seed, weightsBuilder)

}

object MatrixTransformBuilder {

  final def apply()
  : MatrixTransformBuilder = new MatrixTransformBuilder()

  final def apply(matrix: DenseMatrix[Real])
  : MatrixTransformBuilder = apply().setMatrix(matrix)

  /**
    * CCIR-601 Standard Definition TV
    */
  final def bt601_BGR2Y()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      1,
      3,
      Array(0.114f, 0.587f, 0.299f)
    )
  )

  /**
    * CCIR-601 Standard Definition TV
    */
  final def bt601_RGB2Y()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      1,
      3,
      Array(0.299f, 0.587f, 0.114f)
    )
  )

  /**
    * CCIR-601 Standard Definition TV
    */
  final def bt601_BGR2YUV()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(3, 3, Array(
      +0.114f,   +0.587f,   +0.299f,
      +0.436f,   -0.28886f, -0.14713f,
      -0.10001f, -0.51499f, +0.615f
    ))
  )


  /**
    * CCIR-601 Standard Definition TV
    *
    * \begin{bmatrix} Y' \\ U \\ V \end{bmatrix}
    * &amp;=
    * \begin{bmatrix}
    *    0.299   &amp;  0.587   &amp;  0.114 \\
    *   -0.14713 &amp; -0.28886 &amp;  0.436 \\
    *    0.615   &amp; -0.51499 &amp; -0.10001
    * \end{bmatrix}
    * \begin{bmatrix} R \\ G \\ B \end{bmatrix} \\
    *
    * Also see: https://en.wikipedia.org/wiki/YUV
    */
  final def bt601_RGB2YUV()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      3,
      3,
      Array(
        +0.299f,   +0.587f,   +0.114f,
        -0.14713f, -0.28886f, +0.436f,
        +0.615f,   -0.51499f, -0.10001f
      )
    )
  )

  /**
    * CCIR-601 Standard Definition TV
    */
  final def bt601_YUV2BGR()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      3,
      3,
      Array(
        +1.13983f, +0.0f,     +1.0f,
        -0.58060f, -0.39465f, +1.0f,
        +0.0f,     +2.03211f, +1.0f
      )
    )
  )

  /**
    * CCIR-601 Standard Definition TV
    *
    * \begin{bmatrix} R \\ G \\ B \end{bmatrix}
    * &amp;=
    * \begin{bmatrix}
    *   1 &amp;  0       &amp;  1.13983 \\
    *   1 &amp; -0.39465 &amp; -0.58060 \\
    *   1 &amp;  2.03211 &amp;  0
    * \end{bmatrix}
    * \begin{bmatrix} Y' \\ U \\ V \end{bmatrix}
    *
    * Also see: https://en.wikipedia.org/wiki/YUV
    */
  final def bt601_YUV2RGB()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      3,
      3,
      Array(
        +1.0f, +0.0f,     +1.13983f,
        +1.0f, -0.39465f, -0.58060f,
        +1.0f, +2.03211f, +0.0f
      )
    )
  )

  /**
    * CCIR-709 High Definition TV
    */
  final def bt709_BGR2Y()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      1,
      3,
      Array(0.0722f, 0.7152f, 0.2126f)
    )
  )

  /**
    * CCIR-709 High Definition TV
    */
  final def bt709_RGB2Y()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      1,
      3,
      Array(0.2126f, 0.7152f, 0.0722f)
    )
  )

  /**
    * CCIR-709 High Definition TV
    */
  final def bt709_BGR2YUV()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      3,
      3,
      Array(
        +0.0722f,  +0.7152f,  +0.2126f,
        +0.436f,   -0.33609f, -0.09991f,
        -0.05639f, -0.55861f, +0.615f
      )
    )
  )

  /**
    * CCIR-709 High Definition TV
    *
    * \begin{bmatrix} Y' \\ U \\ V \end{bmatrix}
    * &amp;=
    * \begin{bmatrix}
    *    0.2126  &amp;  0.7152  &amp;  0.0722 \\
    *   -0.09991 &amp; -0.33609 &amp;  0.436  \\
    *    0.615   &amp; -0.55861 &amp; -0.05639
    * \end{bmatrix}
    * \begin{bmatrix} R \\ G \\ B \end{bmatrix} \\
    *
    * Also see: https://en.wikipedia.org/wiki/YUV
    */
  final def bt709_RGB2YUV()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      3,
      3,
      Array(
        +0.2126f,  +0.7152f,  +0.0722f,
        -0.09991f, -0.33609f, +0.436f,
        +0.615f,   -0.55861f, -0.05639f
      )
    )
  )

  /**
    * CCIR-709 High Definition TV
    */
  final def bt709_YUV2BGR()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      3,
      3,
      Array(
        +1.28033f, +0.0f,     +1.0f,
        -0.38059f, -0.21482f, +1.0f,
        +0.0f,     +2.12798f, +1.0f
      )
    )
  )

  /**
    * CCIR-709 High Definition TV
    *
    * \begin{bmatrix} R \\ G \\ B \end{bmatrix}
    * &amp;=
    * \begin{bmatrix}
    *   1 &amp;  0       &amp;  1.28033 \\
    *   1 &amp; -0.21482 &amp; -0.38059 \\
    *   1 &amp;  2.12798 &amp;  0
    * \end{bmatrix}
    * \begin{bmatrix} Y' \\ U \\ V \end{bmatrix}
    *
    * Also see: https://en.wikipedia.org/wiki/YUV
    */
  final def bt709_YUV2RGB()
  : MatrixTransformBuilder = apply().setMatrix(
    new DenseMatrix(
      3,
      3,
      Array(
        +1.0f, +0.0f,     +1.28033f,
        +1.0f, -0.21482f, -0.38059f,
        +1.0f, +2.12798f, +0.0f
      )
    )
  )

}
