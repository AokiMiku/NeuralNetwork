using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
	public class NeuroNet
	{
		internal float[][] Neurons { get; private set; }
		internal float[][][] Weights { get; private set; }
		internal float[][] Biases { get; private set; }

		public float[] Outputs
		{
			get
			{
				return this.Neurons[this.Neurons.Length - 1];
			}
		}
		public event EventHandler<FeedForwardFinishedEventArgs> FeedForwardFinished;

		private int[] layers;

		public int LayerCount
		{
			get
			{
				return this.layers.Length;
			}
		}

		private NeuroNet()
		{
			// default ctor
			// used for creating empty networks while crossover
		}

		public NeuroNet(int[] layers)
		{
			this.layers = new int[layers.Length];
			for (int i = 0; i < layers.Length; i++)
			{
				this.layers[i] = layers[i];
			}

			this.InitNeurons();
			this.InitWeightsAndBiases();
		}

		public NeuroNet(NeuroNet copy)
		{
			this.layers = new int[copy.layers.Length];
			for (int i = 0; i < this.layers.Length; i++)
			{
				this.layers[i] = copy.layers[i];
			}

			this.InitNeurons();
			this.InitWeightsAndBiases(copy.Weights, copy.Biases);
		}

		public NeuroNet(Genome genome)
		{
			this.LoadFromGenome(genome);
		}

		private void InitNeurons()
		{
			List<float[]> neurons = new List<float[]>();
			for (int i = 0; i < this.layers.Length; i++)
			{
				float[] layer = new float[this.layers[i]];
				for (int j = 0; j < this.layers[i]; j++)
				{
					layer[j] = 0f;
				}
				neurons.Add(layer);
			}
			this.Neurons = neurons.ToArray();
		}

		private void InitWeightsAndBiases(float[][][] presetWeights = null, float[][] presetBiases = null)
		{
			List<float[][]> weights = new List<float[][]>();
			List<float[]> biases = new List<float[]>();

			for (int i = 1; i < this.layers.Length; i++)
			{
				List<float[]> layerWeights = new List<float[]>();
				float[] layerBiases = new float[this.layers[i]];

				for (int j = 0; j < this.layers[i]; j++)
				{
					float[] neuronWeights = new float[this.layers[i - 1]];
					if (presetBiases != null && presetBiases.Length >= i && presetBiases[i - 1].Length >= j)
					{
						layerBiases[j] = presetBiases[i - 1][j];
					}
					else
					{
						layerBiases[j] = NeuroHelper.RandomNext();
					}

					for (int k = 0; k < neuronWeights.Length; k++)
					{
						if (presetWeights != null && presetWeights.Length >= i && presetWeights[i - 1].Length >= j && presetWeights[i - 1][j].Length >= k)
						{
							neuronWeights[k] = presetWeights[i - 1][j][k];
						}
						else
						{
							neuronWeights[k] = NeuroHelper.RandomNext();
						}
					}
					layerWeights.Add(neuronWeights);
				}
				biases.Add(layerBiases);
				weights.Add(layerWeights.ToArray());
			}

			this.Weights = weights.ToArray();
			this.Biases = biases.ToArray();
		}

		public float[] FeedForward(float[] inputs)
		{
			for (int i = 0; i < inputs.Length; i++)
			{
				this.Neurons[0][i] = inputs[i];
			}

			for (int i = 1; i < this.LayerCount; i++)
			{
				for (int j = 0; j < this.Neurons[i].Length; j++)
				{
					float value = 0f;

					for (int k = 0; k < this.Neurons[i - 1].Length; k++)
					{
						value += this.Weights[i - 1][j][k] * this.Neurons[i - 1][k];
					}

					this.Neurons[i][j] = NeuroHelper.Sigmoid(value);
				}
			}

			this.FeedForwardFinished?.Invoke(this, new FeedForwardFinishedEventArgs(this.Outputs));
			return this.Outputs;
		}

		public void Mutate()
		{
			for (int i = 1; i < this.LayerCount; i++)
			{
				if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.LayerMutationChance)
				{
					List<float[]> neurons = new List<float[]>();
					for (int l = 0; l < this.layers.Length; l++)
					{
						float[] layer = new float[this.layers[l] + 1];
						for (int j = 0; j < this.layers[l]; j++)
						{
							layer[j] = 0f;
						}
						neurons.Add(layer);
					}
					this.Neurons = neurons.ToArray();

					this.layers[i]++; // increase the neuroncount of the layer
					this.InitWeightsAndBiases(this.Weights, this.Biases);
				}

				if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronWeightMutationChance)
				{
					int neuronIndex = NeuroHelper.RandomNext(0, this.Neurons[i].Length);
					int weightIndex = NeuroHelper.RandomNext(0, this.Weights[i][neuronIndex].Length);

					if (weightIndex < this.Weights.Length)
					{
						this.Weights[i][neuronIndex][weightIndex] += NeuroHelper.RandomNext(-NeuroHelper.NeuronWeightMutationDefaultValue, NeuroHelper.NeuronWeightMutationDefaultValue);
					}
				}

				if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronBiasMutationChance)
				{
					int neuronIndex = NeuroHelper.RandomNext(0, this.Neurons[i].Length);

					this.Biases[i][neuronIndex] += NeuroHelper.RandomNext(-NeuroHelper.NeuronBiasMutationDefaultValue, NeuroHelper.NeuronBiasMutationDefaultValue);
				}
			}
		}

		public static NeuroNet Crossover(NeuroNet mother, NeuroNet father)
		{
			NeuroNet child = InitChild(mother, father);

			for (int i = 0; i < child.Weights.Length; i++)
			{
				for (int j = 0; j < child.Weights[i].Length; j++)
				{
					for (int k = 0; k < child.Weights[i][j].Length; k++)
					{
						if (!(k >= mother.Weights[i][j].Length && mother.Weights[i][j].Length < father.Weights[i][j].Length)
							&& ((k >= father.Weights[i][j].Length && mother.Weights[i][j].Length > father.Weights[i][j].Length) || NeuroHelper.RandomNext(0, 100) > 50))
						{
							child.Weights[i][j][k] = mother.Weights[i][j][k];
						}
						else
						{
							child.Weights[i][j][k] = father.Weights[i][j][k];
						}
					}

					if (!(j >= mother.Biases[i].Length && mother.Biases[i].Length < father.Biases[i].Length)
							&& ((j >= father.Biases[i].Length && mother.Biases[i].Length > father.Biases[i].Length) || NeuroHelper.RandomNext(0, 100) > 50))
					{
						child.Biases[i][j] = mother.Biases[i][j];
					}
					else
					{
						child.Biases[i][j] = father.Biases[i][j];
					}
				}
			}

			return child;
		}

		private static NeuroNet InitChild(NeuroNet mother, NeuroNet father)
		{
			int[] fetchedLayers = new int[Math.Max(mother.layers.Length, father.layers.Length)];
			for (int i = 0; i < fetchedLayers.Length; i++)
			{
				if (mother.layers[i] >= father.layers[i])
				{
					fetchedLayers[i] = mother.layers[i];
				}
				else
				{
					fetchedLayers[i] = father.layers[i];
				}
			}

			NeuroNet child = new NeuroNet
			{
				layers = fetchedLayers
			};
			child.InitNeurons();
			child.InitWeightsAndBiases();
			return child;
		}

		private void LoadFromGenome(Genome genome)
		{
			this.layers = new int[genome.LayerCount];
			for (int i = 0; i < this.layers.Length; i++)
			{
				this.layers[i] = genome.NeuronsPerLayer[i];
			}

			this.InitNeurons();
			this.InitWeightsAndBiases();

			int idx = 0;
			for (int i = 0; i < this.LayerCount; i++)
			{
				for (int j = 0; j < this.Weights[i].Length; j++)
				{
					for (int k = 0; k < this.Weights[i][j].Length; k++)
					{
						this.Weights[i][j][k] = genome.Weights[idx];

						idx++;
					}
				}
			}

			idx = 0;
			for (int i = 0; i < this.LayerCount; i++)
			{
				for (int j = 0; j < this.Biases[i].Length; j++)
				{
					this.Biases[i][j] = genome.Biases[idx];

					idx++;
				}
			}
		}
		
		public void Save(string fileName)
		{
			Genome.Save(this, fileName);
		}

		public void Load(string fileName)
		{
			this.LoadFromGenome(Genome.Load(fileName));
		}
	}
}