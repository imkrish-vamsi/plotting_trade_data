# plotting_trade_data
Preparing a dataset manually through data selection on interactive plots.  

PLEASE VISIT: https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/  
In this implementation, fft_size is the number of samples in the fast fourier transform. Setting that value is a tradeoff between the time resolution and frequency resolution you want.  
For example, let’s assume we’re processing a signal with sampling rate of 1000 Hz (and therefore by the Nyqist theorem, a maximum possible recoverable spectrum of 500 Hz). If we choose fft_size = 500, then for each hop, a window of 500 samples will be extracted from the data and turned into 500 individual frequency bins.  
The frequency resolution in this case is 1 Hz, because we have a total possible spectrum of half the sampling rate at 500 Hz, and it’s divided into 500 bins, totaling 1 Hz per bin. The time resolution is 0.5 seconds, because a 500 samples at 1000 Hz is 0.5 seconds. Effectively, a half second of the time domain data has been exchanged for a spectrum with 1 Hz resolution.  
If we choose fft_size = 1000, then we get a worse time resolution of 1 second, but a better frequency resolution of 0.5 Hz.
  
In my implementation, I kept fft_size to powers of 2, because this is the case that the fast fourier transform algorithm is optimized for, but any positive integer can be chosen.
