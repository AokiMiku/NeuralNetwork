using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public class Genome
	{
		public int LayerCount { get; private set; }
		public int[] NeuronsPerLayer { get; private set; }
		public float[] Weights { get; private set; }
		public float[] Biases { get; private set; }

		private const char seperator = ';';

		public Genome(int layerCount, int[] neuronsPerLayer, float[] weights, float[] biases)
		{
			if (neuronsPerLayer == null)
				throw new ArgumentNullException(nameof(neuronsPerLayer));
			if (weights == null)
				throw new ArgumentNullException(nameof(weights));
			if (biases == null)
				throw new ArgumentNullException(nameof(biases));

			LayerCount = layerCount;
			NeuronsPerLayer = neuronsPerLayer;
			Weights = weights;
			Biases = biases;
		}

		public Genome(NeuroNet neuroNet)
		{
			this.LayerCount = neuroNet.LayerCount;

			int[] neuronsPerLayer = new int[LayerCount];
			for (int i = 0; i < neuronsPerLayer.Length; i++)
			{
				neuronsPerLayer[i] = neuroNet.Neurons[i].Length;
			}
			this.NeuronsPerLayer = neuronsPerLayer;

			List<float> Weights = new List<float>();
			for (int i = 1; i < this.LayerCount; i++)
			{
				for (int j = 0; j < neuroNet.Weights[i].Length; j++)
				{
					for (int k = 0; k < neuroNet.Weights[i][j].Length; k++)
					{
						Weights.Add(neuroNet.Weights[i][j][k]);
					}
				}
			}
			this.Weights = Weights.ToArray();

			List<float> Biases = new List<float>();
			for (int i = 1; i < this.LayerCount; i++)
			{
				for (int j = 0; j < neuroNet.Biases[i].Length; j++)
				{
					Biases.Add(neuroNet.Biases[i][j]);
				}
			}
			this.Biases = Biases.ToArray();
		}

		public static void Save(NeuroNet neuroNet, string fileName)
		{
			Genome genome = new Genome(neuroNet);
			string stringifiedGenome = "";

			stringifiedGenome += genome.LayerCount + seperator;
			stringifiedGenome += genome.NeuronsPerLayer.ToString('_') + seperator;
			stringifiedGenome += genome.Weights.ToString('_') + seperator;
			stringifiedGenome += genome.Biases.ToString('_');

			using (StreamWriter streamWriter = new StreamWriter(Environment.CurrentDirectory + "\\Data\\" + fileName + ".aps"))
			{
				streamWriter.Write(stringifiedGenome);
				streamWriter.Flush();
				streamWriter.Close();
			}
		}

		public static Genome Load(string fileName)
		{
			string stringifiedGenome = "";

			using (StreamReader streamReader = new StreamReader(Environment.CurrentDirectory + "\\Data\\" + fileName + ".aps"))
			{
				stringifiedGenome = streamReader.ReadToEnd();
				streamReader.Close();
			}

			int layerCount = int.Parse(stringifiedGenome.Substring(0, stringifiedGenome.IndexOf(seperator)));
			stringifiedGenome = stringifiedGenome.Substring(stringifiedGenome.IndexOf(seperator) + 1);

			List<int> neuronsPerLayer = new List<int>();
			stringifiedGenome = ExtractValues(stringifiedGenome, neuronsPerLayer);

			List<float> weights = new List<float>();
			stringifiedGenome = ExtractValues(stringifiedGenome, weights);

			List<float> biases = new List<float>();
			stringifiedGenome = ExtractValues(stringifiedGenome, biases);

			return new Genome(layerCount, neuronsPerLayer.ToArray(), weights.ToArray(), biases.ToArray());
		}

		private static string ExtractValues(string stringifiedGenome, List<float> listToFill)
		{
			while (stringifiedGenome.IndexOf(seperator) > stringifiedGenome.IndexOf('_'))
			{
				listToFill.Add(float.Parse(stringifiedGenome.Substring(0, stringifiedGenome.IndexOf('_'))));

				stringifiedGenome = stringifiedGenome.Substring(stringifiedGenome.IndexOf('_') + 1);
			}

			if (stringifiedGenome.Contains(seperator))
			{
				stringifiedGenome = stringifiedGenome.Substring(stringifiedGenome.IndexOf(seperator) + 1);
			}
			return stringifiedGenome;
		}

		private static string ExtractValues(string stringifiedGenome, List<int> listToFill)
		{
			List<float> lst = new List<float>();
			stringifiedGenome = ExtractValues(stringifiedGenome, lst);
			for (int i = 0; i < lst.Count; i++)
			{
				listToFill.Add((int)lst[i]);
			}
			return stringifiedGenome;
		}
	}
}