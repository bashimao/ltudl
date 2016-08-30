package edu.latrobe.blaze.modules.jvm

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._

final class AddGaussianNoise_JVM_Baseline(override val builder:        AddGaussianNoiseBuilder,
                                          override val inputHints:     BuildHints,
                                          override val seed:           InstanceSeed,
                                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends AddGaussianNoise_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    ArrayEx.transform(
      output.values
    )(_ + gaussian.sample())
  }

}

object AddGaussianNoise_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[AddGaussianNoiseBuilder] {

  override def build(builder:             AddGaussianNoiseBuilder,
                     hints:               BuildHints,
                     seed:                InstanceSeed,
                     weightBufferBuilder: ValueTensorBufferBuilder)
  : Module = new AddGaussianNoise_JVM_Baseline(
    builder,
    hints,
    seed,
    weightBufferBuilder
  )

}
