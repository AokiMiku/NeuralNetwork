namespace NeuralNetwork
{
	public class NeuroLayer
	{
		public Neuron[] Neurons { get; private set; }
		public float[] Inputs { get; private set; }
		public float[] Outputs { get; private set; }

		private int numberInputs;
		private int numberOutputs;

		public NeuroLayer(int numberInputs, int numberOutputs)
		{
			this.numberInputs = numberInputs;
			this.numberOutputs = numberOutputs;
			this.Neurons = new Neuron[numberOutputs];
			this.Inputs = new float[this.numberInputs];
			this.Outputs = new float[this.numberOutputs];
			this.InitNeurons();
		}

		public NeuroLayer(NeuroLayer copy)
		{
			this.numberInputs = copy.numberInputs;
			this.numberOutputs = copy.numberOutputs;
			this.Neurons = new Neuron[copy.numberOutputs];
			this.Inputs = new float[copy.numberInputs];
			this.Outputs = new float[copy.numberOutputs];
			this.CopyNeurons(copy.Neurons);
		}

		internal void InitNeurons()
		{
			for (int i = 0; i < this.numberOutputs; i++)
			{
				this.Neurons[i] = new Neuron(this.numberInputs);
			}
		}

		private void CopyNeurons(Neuron[] neurons)
		{
			for (int i = 0; i < this.numberOutputs; i++)
			{
				this.Neurons[i] = new Neuron(neurons[i]);
			}
		}

		internal float[] FeedForward(float[] inputs)
		{
			this.CopyInputs(inputs);

			for (int i = 0; i < this.Neurons.Length; i++)
			{
				this.Outputs[i] = this.Neurons[i].FeedForward(inputs);
			}

			return this.Outputs;
		}

		internal float[] FeedForwardInput(float[] inputs)
		{
			this.CopyInputs(inputs);
			this.Outputs = inputs;

			return this.Outputs;
		}

		private void CopyInputs(float[] inputs)
		{
			this.Inputs = new float[inputs.Length];
			for (int i = 0; i < inputs.Length; i++)
			{
				this.Inputs[i] = inputs[i];
			}
		}

		internal void Mutate()
		{
			//if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.LayerMutateChance)
			//{

			//}
			if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronWeightMutationChance)
			{
				UnityEngine.Debug.Log("NeuronWeightMutateChance proc");
				Neuron temp = this.Neurons[NeuroHelper.RandomNext(0, this.Neurons.Length)];
				temp.MutateWeight(NeuroHelper.RandomNext(0, temp.Weights.Length));
			}
			if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronBiasMutationChance)
			{
				UnityEngine.Debug.Log("NeuronBiasMutateChance proc");
				this.Neurons[NeuroHelper.RandomNext(0, this.Neurons.Length)].MutateBias();
			}
		}
	}
}