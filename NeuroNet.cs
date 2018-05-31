namespace NeuralNetwork
{
	public class NeuroNet
	{
		public NeuroLayer[] NeuroLayers { get; private set; }
		public float[] Outputs { get; private set; }

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
	}
}