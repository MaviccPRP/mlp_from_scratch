
## Table of contents

* [Synopsis](#synopsis)
* [Structure and Components](#structure-and-components)
* [Prerequisites](#prerequisites)
* [Quickstart](#quickstart)
* [Experiments and Evaluation](#experiments-and-evaluation)
* [To-Dos](#to-dos)
* [Final Thoughts](#final-thoughts-and-future-work)
* [References](#references)


## Synopsis
To better understand the processes in a multi layer perceptron, this projects implements a simple mlp from scratch using no external machine learning libraries. Algebraic or calculus libraries are just used in a saving manner. 
This is a multi layer perceptron written in Python 3.
## Structure and Components
This project contains three modules:
- mlp_np.py uses NumPy for linear algebra and calculus operations
- mlp_plain.py uses **no** additional libraries in the feed forward and backpropagation process
- algebra_helpers.py contains methods for linear algebra

The mlp consists of an input layer a hidden layer and an output layer. The hidden layer uses a ReLU activation function, sigmoid is available, too. The output layer uses a softmax function to predict a multinomial problem. The input data labels needs to be encoded as one hot vectors. You can find a one hot vector encoder and decoder for multinomial classes in the code.
The mlp has three optimization algorithms implemented:
- Stochastic gradient descent
- Momentum
- AdaGrad

All three optimizers are based on calculating gradients per sample.
The scripts contain the following methods:

- **fit**
Fits the model to the data. If verbose mode, returns the number of number of seen samples as a list and the suitable accuracy scores as a list.
- **predict**
Makes predictions of given dataset samples and labels using the fitted model

The mlp class takes following parameters.
- dataset samples as a list of lists
- dataset labels as a list of lists
- size of the hidden layer (default: 30)
- eta hyperparameter (default: 0.1)
- my hyperparameter (default: 0.9)
- number of epochs (default: 10)
- optimizer type (default: SGD)
- show intermediate steps, including accuracy score per epoch (default: True)

## Prerequisites
- Python 3.4+
	- Scikit Learn zur Evaluation und Datenbeschaffung
    - NumPy
	- matplotlib
## Quickstart
```
$ virtualenv -p python3 venv
$ source venv/bin/activate  
$ pip install -r requirements.txt  
```
Congratulations! You are now ready to train the mlp recognizing handwritten digits.
The scripts by default uses the MNIST dataset for fitting. Generally it can run on any dataset fullfilling the sklearn dataset format.
To run the mlp with NumPy just type in:
```
$ python mlp_np.py
```
To run the plain mlp type:
```
$ python mlp_plain.py
```
If you want to change hyperparameters and optimizers, edit lines XX in the script.
```python
etas = [0.5,0.8,1,2]
for eta in etas:
    NN = NeuralNetwork(samples=X_train, labels=y_train, eta=eta, epochs=50, size_hidden=40, optimizer="sgd", verbose=True)
    fitted = NN.fit()

```
## Experiments and Evaluation
The dataset is splitted into a trainingset (46900 samples) and a testset (23100 samples) using the train_test_split method of sklearn. 
For the evaluation mlp_np.py is used, as it performs much faster, than mlp_plain.py. Seeing 46900 Samples mlp_np.py needs 14 seconds and mlp_plain.py 399 seconds.
The model is evaluated after **2.400.000** samples are seen. The accuracy scores are based on the model performance on the test set.  The Hyperparameters for each optimizer are noted in the table:

| Optimizer  | Best Accuracy score |
| ------------- | ------------- |
| SGD  *Eta=3.5*|  97,1% |
| Momentum *Eta=0.1, My=0.9*| 96,5%  |
| AdaGrad *Eta=1*| 96,3%|

![Evaluation Curve - Accuracy vs Samples seen](https://github.com/MaviccPRP/mlp_from_scratch/blob/master/eval_nn.png)
## To-Dos
- [X] Implement per sample SGD
- [ ] Implementing mini-batch 
- [ ] Implement the AdaGrad with diagonal matrices
- [ ] Gridsearch for optimizing hyperparameters
- [ ] Compare evaluation performance on testset and trainingset

## Final Thoughts and Future Work

- Best results are reached using the SGD per sample algorithm. 
- Momentum is oscillating, when higher percentages are reached. This could be because of the per sample approach. 
- The AdaGrad algorithms is smoother than Momentum and SGD, but does not reach accuracy scores of SGD or Momentum.
- In future work, implementing mini-natches could help improving accuracy performance and smoothness of Momentum and AdaGrad. 
- Still hyperparameters are not fully evaluated, this could be done using Gridsearch.
- Additionally the AdaGrad algorithm could be updated with a diagonal matrix approach.
- Comparing accuracy on test and training sets, to exclude overfitting.
## References
- Principles of training multi-layer neural network using backpropagation, http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
- A Step by Step Backpropagation Example, https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
- Julia Kreutzer, Julian Hitschler, Neural Networks: Architectures and Applications for NLP, http://www.cl.uni-heidelberg.de/courses/ws16/neuralnetworks/slides/session02.pdf
- Julia Kreutzer, Julian Hitschler, Neural Networks: Architectures and Applications for NLP, http://www.cl.uni-heidelberg.de/courses/ws16/neuralnetworks/slides/session03.pdf
- A Neural Network in 13 lines of Python (Part 2 - Gradient Descent), http://iamtrask.github.io/2015/07/27/python-network-part2/
- How the backpropagation algorithm works, http://neuralnetworksanddeeplearning.com/chap2.html
- Paul Fackler, Notes on Matrix Calculus, http://www4.ncsu.edu/%7Epfackler/MatCalc.pdf
