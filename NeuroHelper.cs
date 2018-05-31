using System;

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
			return 2f / (1 + (float)Math.Exp(-2f * value)) - 1f;
		}
	}
}