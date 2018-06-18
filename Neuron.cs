using System.Linq;

namespace NeuralNetwork
{
	internal class Neuron
	{
		public float[] Weights { get; private set; }
		public float Bias { get; private set; }

		public Neuron(int neuronsInPrevLayer)
		{
			this.Weights = new float[neuronsInPrevLayer];
			this.InitWeights();
			this.InitBias();
		}

		public Neuron(Neuron copy)
		{
			this.Weights = new float[copy.Weights.Length];
			this.InitWeights(copy.Weights);
			this.InitBias(copy.Bias);
		}

		internal void InitWeights()
		{
			for (int i = 0; i < this.Weights.Length; i++)
			{
				this.Weights[i] = NeuroHelper.RandomNext();
			}
		}

		private void InitWeights(float[] weights)
		{
			if (this.Weights.Length != weights.Length)
			{
				return;
			}

			for (int i = 0; i < this.Weights.Length; i++)
			{
				this.Weights[i] = weights[i];
			}
		}

		internal void InitBias()
		{
			this.Bias = NeuroHelper.RandomNext();
		}

		private void InitBias(float bias)
		{
			this.Bias = bias;
		}

		internal float FeedForward(float[] inputs)
		{
			float output = 0;

			for (int i = 0; i < inputs.Length; i++)
			{
				output += this.Weights[i] * inputs[i];
			}

			return NeuroHelper.Sigmoid(output) + this.Bias;
		}

		internal float FeedForwardInput(float input)
		{
			return input;
		}

		internal void MutateWeight(int weightIndex, float mutateAmount = NeuroHelper.NeuronWeightMutationDefaultValue)
		{
			if (weightIndex > this.Weights.Length)
			{
				return;
			}

			this.Weights[weightIndex] += NeuroHelper.RandomNext(-mutateAmount, mutateAmount);
		}

		internal void MutateBias(float mutateAmount = NeuroHelper.NeuronBiasMutationDefaultValue)
		{
			this.Bias += NeuroHelper.RandomNext(-mutateAmount, mutateAmount);
		}
	}
}