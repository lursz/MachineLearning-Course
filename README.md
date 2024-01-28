# MachineLearningCourse
This repository contains full Machine Learning course, held on the Computer Science and Intelligent Systems, AGH.

## Table of contents
| Description  | Lecture | Assignment |
| ------------- | ------------- | ------------- |
| 01. Introduction  | [lecture1](Lectures/01-wprowadzenie.pdf)  | [lab1](lab01_introduction/) | 
| 02. Classification  | [lecture2](Lectures/02-klasyfikacja.pdf)  | [lab2](lab02_classification/) |
| 03. Regression  | [lecture3](Lectures/03-regresja.pdf)  | [lab3](lab03_regression/) |
| 04. SVM (Support vector machine)  | [lecture4](Lectures/04-svm.pdf)  | [lab4](lab04_svm/) |
| 05. Trees  | [lecture5](Lectures/05-drzewa.pdf)  | [lab5](lab05_trees/) |
| 06. Ensemble  | [lecture6](Lectures/06-ensemble.pdf) | [lab6](lab06_ensemble/) |
| 07. Clustering  | [lecture7](Lectures/07-nienadzorowane.pdf)  | [lab7](lab07_clustering/) |
| 08. Scaling  | [lecture8](Lectures/08-redukcja_wymiarow.pdf)  | [lab8](lab08_scaling/) |
| 09. Neural Networks  | [lecture9](Lectures/09-sieci-neuronowe-latest.pdf)  | [lab9](lab09_neural-networks/) |
| 10. Neural Networks Keras  | [lecture10](Lectures/10-sieci-neuronowe-cd-latest.pdf)  | [lab10](lab10_neural-networks-keras/) |
| 11. Hyperparameters | [lecture11](Lectures/11-uczenie-sieci.pdf)  | [lab11](lab11_hyperparameters/) | 
| 12. Convolutional Neural Networks  | [lecture12](Lectures/12-cnn-latest.pdf)  | [lab12](lab12_CNN/) | 
| 13. Recurrent Neural Network  | [lecture13](Lectures/13-rnn-latest.pdf)  | [lab13](lab13_RNN/) | 
| 14. Autoencoders  | [lecture14](Lectures/14-autoenkodery-gan-latest.pdf)  | - |


## Some Notes

<details>
<summary>SVM (Support vector machine)</summary>

- Not all datapoints are linearly separable on lower dimension
- Transform such dataset to a higher dimensional space where it can be linearly separable by a hyperplane
- Support vectors: 
    - examples/data points closest to the hyperplane
    - both classified and misclassified datapoints are counted
    - If a datapoint is not a support vector, removing it will not affect the model
    - Small number of support vectors = fast kernel SVMs
- margin : distance from a support vector to decision boundary
- best decision boundary has equal distance from all support vectors
- The best separable line is the hyperplane has the biggest margin
- Measure of closeness : Regularization parameters (hinge loss and l2)
- Boundary decision lines : The lines that touches the support vectors / closest the the support vectors
- Soft margin SVM : Used when the classes are not separable (Controlled by regularization parameter)
- kernel : Sometimes it is difficult to caculate the mapping of transformation. So we use a shortcut called kernel that is computationally less expensive.
    - RBF : support vector = Difference between 2 inputs: X and X` 
    - C hyperparameter = regularization
    - gamma hyperparameter = smoothness of the boundary
- Stochastic gradient descent (SGD) : 
    - Similar to SVM, but scales well for large dataset
    - how : uses gradient descent to find out the maximised margin among possible margins.
</details>

<details>
<summary>Trees </summary>
	
- Sequence of if-else question
- Consists of hierarchy of nodes. Each node raise question or prediction.
- Root node : No parent
- Internal node : Has parent, has children
- Leaf node : Has no children. It is where predictions are made
- Goal : Search for pattern to produce purest leaves. Each leaf contains pattern for one dominant label.
- Information Gain : At each node, find the split point for each feature for which we get maximum correct pure split of the data. When information gain = 0, we could say that our goal is achieved, the pattern is captured, and this is a leaf node. Otherwise keep splitting it (We can stop it by specifying maximum depth of recursion split). 
- Measure of impurity in a node:
    - Gini index: For classification
    - Entropy: For classification
    - MSE : For regression
- capture non-linear relationhship between features and labels/ real values
- Do not require feature scaling
- At each split, only one feature is involved
- Decision region : Feature space where instances are assigned to a label / value
- Decision Boundary : Surface that separates different decision regions
- Steps of building a decision tree:
    1. Choose an attribute (column) of dataset
    2. Calculate the significance of that attribute when splitting the data with Entropy.
        A good split has less Entropy (disorder / randomness). 
    3. Find the best attribute that has most significance and use that attribute
    	to split the data
    4. For each branch, repeat the process (Recursive partitioning) for best 
    	information gain (The path that gives the most information using entropy).
- Limitations:
    - Can only produce orthogonal decision boundaries
    - Sensitive to small variations in training set
    - High variance overfits the model
- Solution : Ensemble learning
    - Train different models on same dataset
    - Let each model make its prediction
    - Aggregate predictions of individual models (eg: hard-voting)
    - One model's weakness is covered by another model's strength in that particular task
    - Final model is combination of models that are skillfull in different ways
</details>

<details>
<summary>Ensemble Learning</summary>
	
- Limitations of simple decision tree:
    - Can only produce orthogonal decision boundaries
    - Sensitive to small variations in training set
    - High variance overfits the model
- Solution : Ensemble learning
    - This is a joint modeling where many models come together to solve a single problem
    - Train different models on same dataset
    - Let each model make its prediction
    - Aggregate predictions of individual models 
    - One model's weakness is covered by another model's strength in that particular task
    - Final model is combination of models that are skillfull in different ways
    - Hard-voting : 
        - Ensemble method that models data using majority of vote
    - Bagging or Bootstrap aggregating (Sampling with replacement) : 
        - Ensemble method that use bootstrap with resampling on training data. 
        - Base estimator : Decision tree, neural net, logistic regression etc
        - Reduces variance in individual models (Because of bootstrapping, variance of sample becomes smaller)
        - OOB evaluation : normally on average 33% sample data remains unseen, use that data for evaluation of scoring
        - Classification : Final prediction is obtained by majority voting
        - Regression : Final prediction is obtained by taking the mean
    - Random Forest (Sampling without replacement):
        - base estimator : Decision tree
        - bootstrap samples without replacement and further randomization involved
        - Classification : Final prediction is obtained by majority voting
        - Regression : Final prediction is obtained by taking the mean
    - Boosting:
        - Combine weak learners (models that are slightly better than random guessing) to form a strong learner
        - learners are placed sequentially, each learner trying to correct its predecessor
        - Adaboost or adaptive boosting (through contribution/weight adjustment) : 
            - predictor pays more information to wrongly classified target by predecessor and apply a weight or penalty
            - each predictor has an assigned co-efficient (alpha), that signifies it's contribution in final prediction
            - before the data goes to the next predictor for training, alpha is used to adjust the weights of data 
            - Learning rate ita contributes the adjustment of co-efficient alpha
            - Classification : Final outcome decided by weighted majority voting
            - Regression : Final outcome decided by weighted average
        - Gradient boosting (through training on gradients/residuals) :
            - sequential correction of predecessor's error instead of co-efficient adjustment like adaboost
            - Instead of adjusting weight like adaboost, predictor trains using predecessor's residuals as labels
            - Instead of weak learner like adaboost, it uses CART learners as base learners
            - Learning rate or shrinkage tradeoff : Decreased learning rate = increased number of estimators
        - Stochastic gradient boosting (sampling without replacement on gradient boosting to increase variance)
            - Gradient boosting problem : May lead to CARTs using the same split points and maybe the same features which may lead to increased bias. This may lead to underfitting problem.
            - Goal : to reduce bias and increase variance.
</details>

<details>
<summary>Neural Networks</summary>
	
Step by step:

There is a matrix of nodes like this:
```
0 0 0
0 0 0
0 0 0
```
Inputs are fed in from the left. like this:
```
(.1) 0 0
(.24) 0 0		(inputs in parenthesis)
(0) 0 0
```
the next nodes data is calculated by multiplying A weight to each node 
in the layer before and then adding bias. And then putting in an 
activation function. Like this:
```
.1  ----(weight: 1)--> .1
.24 ----(weight: .5)--> .12  ----> .1 + .12 + 0 = .22 + bias
0   ----(weight: 2)-->  0
```

Each node has a bias, for this node lets say its .3
```
.22 + .3(bias) = .52
```
After all that you put it in an activation function.
```
tanh(.52) ---> data for the next node
```
Everything was just the caclulation for the 
first node in the second layer though so the matrix
would look like this
after all that.
```
.1 tanh(.52) 0
.24 0 0
0 0 0
```
Then you keep doing that for each layer until you reach the end.

(Q A)

Does each node have a weight?

No. Each connection has a weight. Or i guess you could say each node
has lots of weights each corresponding to a node in the last layer.


Does each node have a bias?
Yes each node has only one bias.

</details>


<details>
<summary>Convolutional Neural Networks</summary>

- Similar to neural networks
    - normal neural networks take N X 1  inputs for N no of columns
    - CNNs take R X C X N  for R no of rows, C no of columns and N no of 
    	channels in an image
- Takes inputs as images
- Allows us to incorporate certain properties into the architecture for images
    - Smooth forward propagation
- Reduced parameters
    - Convolution helps us to reduce parameters and fasten computation 
    - Helps us to retain special dimensions and informations
- Working Process:
    1. Convolution applies filters to sort out special dimensions
    2. Pooling helps to extract significant pattern in the spatial dimension
    3. Fully connected layer flattens the last Convolution or Pooling layer 
    	and connect with all the nodes of the flattenned layer with all the 
        nodes in output layer in a dense manner

### Recurrent Neural Network
- Traditional neural networks take independent scenes as inputs
- Recurrent neural networks incorporates dependency of 
	sequence within a neural network
- RNNs are networks with loops
- All nodes compute: 
	input_data * input_weight + previous_node_output * node_weight = new_output

</details>
