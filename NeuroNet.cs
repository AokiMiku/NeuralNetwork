namespace NeuralNetwork
{
	public class NeuroNet
	{
		internal NeuroLayer[] NeuroLayers { get; private set; }
		public float[] Outputs { get; private set; }

		public int LayerCount
		{
			get
			{
				return this.NeuroLayers.Length;
			}
		}

		private NeuroNet()
		{

		}

		public NeuroNet(int[] layers)
		{
			this.NeuroLayers = new NeuroLayer[layers.Length];
			this.NeuroLayers[0] = new NeuroLayer(0, layers[0]);
			for (int i = 1; i < layers.Length; i++)
			{
				this.NeuroLayers[i] = new NeuroLayer(layers[i - 1], layers[i]);
			}
		}

		public NeuroNet(NeuroNet copy)
		{
			this.NeuroLayers = new NeuroLayer[copy.NeuroLayers.Length];
			for (int i = 0; i < this.NeuroLayers.Length; i++)
			{
				this.NeuroLayers[i] = new NeuroLayer(copy.NeuroLayers[i]);
			}
		}

		public float[] FeedForward(float[] inputs)
		{
			this.Outputs = this.NeuroLayers[0].FeedForwardInput(inputs);
			for (int i = 1; i < this.NeuroLayers.Length; i++)
			{
				this.Outputs = this.NeuroLayers[i].FeedForward(this.Outputs);
			}

			return this.Outputs;
		}

		public void Mutate()
		{
			for (int i = 1; i < this.NeuroLayers.Length; i++)
			{
				this.NeuroLayers[i].Mutate();
			}
		}

		public static NeuroNet Crossover(NeuroNet mother, NeuroNet father)
		{
			NeuroNet child = new NeuroNet
			{
				NeuroLayers = new NeuroLayer[mother.LayerCount]
			};

			for (int i = 0; i < mother.LayerCount; i++)
			{
				child.NeuroLayers[i] = new NeuroLayer(mother.NeuroLayers[i].Inputs.Length, mother.NeuroLayers[i].Outputs.Length);
				for (int j = 0; j < mother.NeuroLayers.Length; j++)
				{
					if (NeuroHelper.RandomNext(0, 1f) < 0.5f)
					{
						child.NeuroLayers[i].Neurons[j] = new Neuron(mother.NeuroLayers[i].Neurons[j]);
					}
					else
					{
						child.NeuroLayers[i].Neurons[j] = new Neuron(father.NeuroLayers[i].Neurons[j]);
					}
				}
			}

			return child;
		}
	}
}