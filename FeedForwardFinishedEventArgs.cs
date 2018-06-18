using System;

namespace NeuralNetwork
{
	public class FeedForwardFinishedEventArgs : EventArgs
	{
		public float[] Outputs { get; private set; }

		public FeedForwardFinishedEventArgs(float[] outputs)
		{
			Outputs = outputs;
		}
	}
}
