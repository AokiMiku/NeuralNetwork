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
				neurons.Add(new float[this.layers[i]]);
			}

			this.Neurons = neurons.ToArray();
		}

		private void InitWeightsAndBiases()
		{
			List<float[][]> weights = new List<float[][]>();
			List<float[]> biases = new List<float[]>();

			for (int i = 1; i < this.layers.Length; i++)
			{
				List<float[]> layerWeights = new List<float[]>();
				int neuronsInPreviousLayer = this.layers[i - 1];
				List<float> layerBiases = new List<float>();

				for (int j = 0; j < this.Neurons[i].Length; j++)
				{
					float[] neuronWeights = new float[neuronsInPreviousLayer];
					layerBiases.Add(NeuroHelper.RandomNext());

					for (int k = 0; k < neuronsInPreviousLayer; k++)
					{
						neuronWeights[k] = NeuroHelper.RandomNext();
					}

					layerWeights.Add(neuronWeights);
				}

				weights.Add(layerWeights.ToArray());
				biases.Add(layerBiases.ToArray());
			}

			this.Weights = weights.ToArray();
			this.Biases = biases.ToArray();
		}

		private void InitWeightsAndBiases(float[][][] weights, float[][] biases)
		{
			List<float[][]> weightsList = new List<float[][]>();
			List<float[]> biasesList = new List<float[]>();

			for (int i = 0; i < weights.Length; i++)
			{
				List<float[]> layerWeights = new List<float[]>();
				float[] layerBiases = new float[biases[i].Length];

				for (int j = 0; j < weights[i].Length; j++)
				{
					float[] neuronWeights = new float[weights[i][j].Length];

					for (int k = 0; k < weights[i][j].Length; k++)
					{
						neuronWeights[k] = weights[i][j][k];
					}

					layerWeights.Add(neuronWeights);
					layerBiases[j] = biases[i][j];
				}
				weightsList.Add(layerWeights.ToArray());
				biasesList.Add(layerBiases);
			}

			this.Weights = weightsList.ToArray();
			this.Biases = biasesList.ToArray();
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
				//if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.LayerMutateChance)
				//{

				//}

				if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronWeightMutationChance)
				{
					UnityEngine.Debug.Log("NeuronWeightMutateChance proc");
					int neuronIndex = NeuroHelper.RandomNext(0, this.Neurons[i].Length);
					int weightIndex = NeuroHelper.RandomNext(0, this.Weights[i][neuronIndex].Length);

					if (weightIndex < this.Weights.Length)
					{
						this.Weights[i][neuronIndex][weightIndex] += NeuroHelper.RandomNext(-NeuroHelper.NeuronWeightMutationDefaultValue, NeuroHelper.NeuronWeightMutationDefaultValue);
					}
				}

				if (NeuroHelper.RandomNext(0f, 1f) <= NeuroHelper.NeuronBiasMutationChance)
				{
					UnityEngine.Debug.Log("NeuronBiasMutateChance proc");
					int neuronIndex = NeuroHelper.RandomNext(0, this.Neurons[i].Length);

					this.Biases[i][neuronIndex] += NeuroHelper.RandomNext(-NeuroHelper.NeuronBiasMutationDefaultValue, NeuroHelper.NeuronBiasMutationDefaultValue);
				}
			}
		}

		public static NeuroNet Crossover(NeuroNet mother, NeuroNet father)
		{
			NeuroNet child = new NeuroNet
			{
				layers = mother.layers
			};
			child.InitNeurons();
			child.InitWeightsAndBiases();

			for (int i = 0; i < mother.layers.Length; i++)
			{
				for (int j = 0; j < mother.Neurons[i].Length; j++)
				{
					for (int k = 0; k < mother.Weights[i][j].Length; k++)
					{
						if (NeuroHelper.RandomNext(0, 100) > 50)
						{
							child.Weights[i][j][k] = mother.Weights[i][j][k];
							child.Biases[i][j] = mother.Biases[i][j];
						}
						else
						{
							child.Weights[i][j][k] = father.Weights[i][j][k];
							child.Biases[i][j] = father.Biases[i][j];
						}
					}
				}
			}

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