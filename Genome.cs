using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	internal class Genome
	{
		public int LayerCount { get; private set; }
		public int[] NeuronsPerLayer { get; private set; }
		public float[] Weights { get; private set; }
		public float[] Biases { get; private set; }

		public Genome(int layerCount, int[] neuronsPerLayer, float[] weights, float[] biases)
		{
			LayerCount = layerCount;
			NeuronsPerLayer = neuronsPerLayer ?? throw new ArgumentNullException(nameof(neuronsPerLayer));
			Weights = weights ?? throw new ArgumentNullException(nameof(weights));
			Biases = biases ?? throw new ArgumentNullException(nameof(biases));
		}

		public Genome(NeuroNet neuroNet)
		{
			this.LayerCount = neuroNet.LayerCount;

			int[] neuronsPerLayer = new int[LayerCount];
			for (int i = 0; i < neuronsPerLayer.Length; i++)
			{
				neuronsPerLayer[i] = neuroNet.NeuroLayers[i].Neurons.Length;
			}

			List<float> Weights = new List<float>();
			for (int i = 1; i < this.LayerCount; i++)
			{
				for (int j = 0; j < neuroNet.NeuroLayers[i].Neurons.Length; j++)
				{
					for (int k = 0; k < neuroNet.NeuroLayers[i].Neurons[j].Weights.Length; k++)
					{
						Weights.Add(neuroNet.NeuroLayers[i].Neurons[j].Weights[k]);
					}
				}
			}
			this.Weights = Weights.ToArray();

			List<float> Biases = new List<float>();
			for (int i = 1; i < this.LayerCount; i++)
			{
				for (int j = 0; j < neuroNet.NeuroLayers[i].Neurons.Length; j++)
				{
					Biases.Add(neuroNet.NeuroLayers[i].Neurons[j].Bias);
				}
			}
			this.Biases = Biases.ToArray();
		}

		public static void Save(Genome genome, string fileName)
		{
			string stringifiedGenome = "";

			stringifiedGenome += genome.LayerCount + "-";
			stringifiedGenome += genome.NeuronsPerLayer.ToString('_') + "-";
			stringifiedGenome += genome.Weights.ToString('_') + "-";
			stringifiedGenome += genome.Biases.ToString('_');

			return;
		}

		public static Genome Load(string fileName)
		{
			string stringifiedGenome = "";

			int layerCount = int.Parse(stringifiedGenome.Substring(0, stringifiedGenome.IndexOf('-')));
		}
	}
}