# Aggressive-Sampling-for-Multi-class-to-BinaryReduction

## Introduction: 

We address the problem of multi-class classification in the case where the number of classes is very large. 
We propose a double sampling strategy on top of a multi-class to binary reduction strategy, which transforms 
the original multi-class problem into a binary classification problem over pairs of examples. 
The aim of the sampling strategy is to overcome the curse of long-tailed class distributions exhibited in majority 
of  large-scale  multi-class classification problems and to reduce the number of pairs of examples in the expanded data. 
Experiments are carried out on DMOZ and Wikipedia collections with 10,000 to 100,000 classes 
where we show the efficiency of the proposed approach in terms of training and prediction time, memory consumption, 
and predictive performance with respect to state-of-the-art approaches.

## Running Instructions: 

python3 run_script_m2b_github_probs.py  [train filename] [test filename] [example_samples] [class_sampling_rate] [Candidates]

Where,

example_samples (mu): Average number of examples sampled per class. The examples are chosen at random from each class with probability based on the distribution. ( e.g. values 1, 2, 5, 10)

class_sampling_rate: Sampling rate for choosing classes to sample ( e.g. 0.1, 0.01, 0.001) (Note: The minimum value for class_
sampling is set as 1 / Size of class, if user enters less than this value by default 1 class will be chosen.

Candidates (sigma): Number of candidate classes for prediction (e.g. 10, 25, 50)

## Link to Datasets:

https://drive.google.com/open?id=0B1ZO0ijiC7SYUk9ybndPbWlCMXM

## Author Information:
1.) Bikash Joshi

2.) Ioannis Partalas

## License: 
This software is licensed under GNU General Public License version 3.0

## Reference: 
https://arxiv.org/pdf/1701.06511.pdf
If you publish results using this program, please acknowledge its use, by referring to: 
Joshi B., Amini M.-R., Partalas I., Iutzeler F., Maximov Y. Aggressive Sampling for Multi-class to Binary Reduction with Applications to Text Classification Advances in Neural Information Processing Systems 30 (NIPS 2017), p. 4162-4171
