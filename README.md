# First NLP Kaggle project

## Intro | Objectives | Main goals
Natural Language Processing (NLP) is a critical area of artificial intelligence (AI) that enables computers to understand, interpret, and respond to human language. Therefore, this project has the goal of studying the Kaggle competition "Natural Language Processing with Disaster Tweets" as way of understanding more about NLP.

## Studying the data
To solve this problem, as a common data science project, it is essential to understand the data. As a NLP problem, the data has to be cleaned and processed to get studied. With the cleaning of words 'http', '@user', etc.

Then, the following steps were as listed:

1. Plot word-clouds to get as much information to get a visual representation of real disaster tweets and the other tweets.
1. Get the top 10 words of each group.
1. Plot the most common keywords for the 


## Building the DL model
After sudying the problem's DF it was essential to start building the DL model from the information presented in the last section. Tensorflow and Keras was used to build this model, as Keras provides an user-friendly interface to work with TensorFlow.

- Using padding as it is common in NLP DL models to get a uniform input size, efficient batch processing, and compatibility with libraries.



### ReLU ACTIVATION 
Rectifier 

$$
f(x)=x^+=max(0,x)=\frac{x+|x|}{2} = \begin{cases}
    x, & \text{if } x > 0 \\
    0, & \text{otherwise}
\end{cases}
$$

$$ReLU(x) = max(0,) $$


### LSTM 
