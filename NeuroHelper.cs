using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
	internal static class NeuroHelper
	{
		private static Random Random = new Random(DateTime.Today.Millisecond);

		public const float NeuronWeightMutationChance = 0.25f;
		public const float NeuronBiasMutationChance = 0.1f;
		public const float LayerMutationChance = 0.001f;
		public const float NeuronWeightMutationDefaultValue = 0.2f;
		public const float NeuronBiasMutationDefaultValue = 0.2f;

		public static float RandomNext(float min = -1f, float max = 1f)
		{
			return Random.Next((int)(min * 1000f), (int)(max * 1000f)) / 1000f;
		}

		public static int RandomNext(int min, int max)
		{
			return Random.Next(min, max);
		}

		public static float Sigmoid(float value)
		{
			return (2f / (1 + (float)Math.Exp(-2f * value))) - 1f;
		}
		public static string ToString(this float[] value, char seperator)
		{
			string s = "";
			for (int i = 0; i < value.Length; i++)
			{
				s += value + seperator.ToString();
			}
			s = s.Substring(0, s.Length - 1);

			return s;
		}

		public static string ToString(this int[] value, char seperator)
		{ 
			float[] f = new float[value.Length];
			value.CopyTo(f, 0);
			return f.ToString(seperator);
		}

		public static int[] ToIntArray(this string value, char seperator)
		{
			List<int> values = new List<int>();
			string s = value;

			for(int i= 0; i < s.Count(x => x == seperator); i++)
			{
				values.Add(int.Parse(s.Substring(0, s.IndexOf(seperator))));

				s = s.Substring(s.IndexOf(seperator) + 1);
			}

			return values.ToArray();
		}

		public static float[] ToFloatArray(this string value, char seperator)
		{
			List<float> values = new List<float>();
			string s = value;

			for (int i = 0; i < s.Count(x => x == seperator); i++)
			{
				if (s.Contains(seperator))
				{
					values.Add(float.Parse(s.Substring(0, s.IndexOf(seperator))));

					s = s.Substring(s.IndexOf(seperator) + 1);
				}
			}

			return values.ToArray();
		}
	}
}