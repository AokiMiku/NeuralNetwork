using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
	public class NeuroNet
	{
		internal List<List<float>> Neurons { get; private set; }
		internal List<List<List<float>>> Weights { get; private set; }
		internal List<List<float>> Biases { get; private set; }

		public List<float> Outputs
		{
			get
			{
				return this.Neurons[this.Neurons.Count - 1];
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
			this.Neurons = new List<List<float>>();
			for (int i = 0; i < this.layers.Length; i++)
			{
				this.Neurons.Add(new List<float>());
				for (int j = 0; j < this.layers[i]; j++)
				{
					this.Neurons[i].Add(0f);
				}
			}
		}

		private void InitWeightsAndBiases()
		{
			this.Weights = new List<List<List<float>>>();
			this.Biases = new List<List<float>>();

			for (int i = 1; i < layers.Length; i++)
			{
				int index = i - 1;
				int neuronsInPrevLayer = layers[index];
				this.Weights.Add(new List<List<float>>());
				this.Biases.Add(new List<float>());

				for (int j = 0; j < layers[i]; j++)
				{
					this.Weights[index].Add(new List<float>());
					this.Biases[index].Add(NeuroHelper.RandomNext());

					for (int k = 0; k < neuronsInPrevLayer; k++)
					{
						this.Weights[index][j].Add(NeuroHelper.RandomNext());
					}
				}
			}
		}

		private void InitWeightsAndBiases(List<List<List<float>>> weights, List<List<float>> biases)
		{
			this.Weights = new List<List<List<float>>>();
			this.Biases = new List<List<float>>();

			for (int i = 0; i < weights.Count; i++)
			{
				this.Weights.Add(new List<List<float>>());
				this.Biases.Add(new List<float>());

				for (int j = 0; j < weights[i].Count; j++)
				{
					this.Weights[i].Add(new List<float>());
					this.Biases[i].Add(biases[i][j]);

					for (int k = 0; k < weights[i][j].Count; k++)
					{
						this.Weights[i][j].Add(weights[i][j][k]);
					}
				}
			}
		}

		public List<float> FeedForward(float[] inputs)
		{
			for (int i = 0; i < inputs.Length; i++)
			{
				this.Neurons[0][i] = inputs[i];
			}

			for (int i = 1; i < this.LayerCount; i++)
			{
				for (int j = 0; j < this.Neurons[i].Count; j++)
				{
					float value = 0f;

					for (int k = 0; k < this.Neurons[i - 1].Count; k++)
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
					this.layers[i]++; // increase the neuroncount of the layer
					this.Neurons[i].Add(0f); // add the new neuron
					this.Biases[i].Add(NeuroHelper.RandomNext()); // add a new bias for the new neuron
					this.Weights[i].Add(new List<float>());

					for (int j = 0; j < this.Weights[i - 1].Count; j++) // add new weight from each neuron in previous layer to the new mutated neuron
					{
						this.Weights[i][this.Weights[i].Count].Add(NeuroHelper.RandomNext()); // only add to the last neuron-weights-list
					}

					if (i < this.LayerCount - 1) // if not output-layer
					{
						for (int j = 0; j < this.Weights[i + 1].Count; j++) // add new weight to each neruon in next layer from the new mutated neuron
						{
							this.Weights[i + 1][j].Add(NeuroHelper.RandomNext());
						}
					}
				}

				if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronWeightMutationChance)
				{
					int neuronIndex = NeuroHelper.RandomNext(0, this.Neurons[i].Count);
					int weightIndex = NeuroHelper.RandomNext(0, this.Weights[i][neuronIndex].Count);

					if (weightIndex < this.Weights.Count)
					{
						this.Weights[i][neuronIndex][weightIndex] += NeuroHelper.RandomNext(-NeuroHelper.NeuronWeightMutationDefaultValue, NeuroHelper.NeuronWeightMutationDefaultValue);
					}
				}

				if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronBiasMutationChance)
				{
					int neuronIndex = NeuroHelper.RandomNext(0, this.Neurons[i].Count);

					this.Biases[i][neuronIndex] += NeuroHelper.RandomNext(-NeuroHelper.NeuronBiasMutationDefaultValue, NeuroHelper.NeuronBiasMutationDefaultValue);
				}
			}
		}

		public static NeuroNet Crossover(NeuroNet mother, NeuroNet father)
		{
			NeuroNet child = InitChild(mother, father);

			for (int i = 0; i < child.Weights.Count; i++)
			{
				for (int j = 0; j < child.Weights[i].Count; j++)
				{
					for (int k = 0; k < child.Weights[i][j].Count; k++)
					{
						if (!(k >= mother.Weights[i][j].Count && mother.Weights[i][j].Count < father.Weights[i][j].Count)
							&& ((k >= father.Weights[i][j].Count && mother.Weights[i][j].Count > father.Weights[i][j].Count) || NeuroHelper.RandomNext(0, 100) > 50))
						{
							child.Weights[i][j][k] = mother.Weights[i][j][k];
						}
						else
						{
							child.Weights[i][j][k] = father.Weights[i][j][k];
						}
					}

					if (!(j >= mother.Biases[i].Count && mother.Biases[i].Count < father.Biases[i].Count)
							&& ((j >= father.Biases[i].Count && mother.Biases[i].Count > father.Biases[i].Count) || NeuroHelper.RandomNext(0, 100) > 50))
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
				for (int j = 0; j < this.Weights[i].Count; j++)
				{
					for (int k = 0; k < this.Weights[i][j].Count; k++)
					{
						this.Weights[i][j][k] = genome.Weights[idx];

						idx++;
					}
				}
			}

			idx = 0;
			for (int i = 0; i < this.LayerCount; i++)
			{
				for (int j = 0; j < this.Biases[i].Count; j++)
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