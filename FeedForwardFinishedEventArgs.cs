using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public class FeedForwardFinishedEventArgs : EventArgs
	{
		public float[] Outputs { get; private set; }

		public FeedForwardFinishedEventArgs(float[] outputs)
		{
			Outputs = outputs;
		}

		public FeedForwardFinishedEventArgs(List<float> outputs)
		{
			Outputs = outputs.ToArray();
		}
	}
}
