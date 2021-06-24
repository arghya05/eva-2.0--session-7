
### Submission by:
1. Sachin Dangayach (sachin.dangayach@gmail.com)

# Objective

Assignment has got 3 parts as mentioned below along with respective solutions:

## 1. Part 1: Submit the Assignment 5 again as mentioned below.

  - Only use datasetSentences.txt. (no augmentation required)
  - Your dataset must have around 12k examples.
  - Split Dataset into 70/30 Train and Test (no validation)
  - Convert floating-point labels into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)

## Solution:

***[Link for colab file for Session7 part 1 submission](https://colab.research.google.com/drive/1B6LeAUgkW0q90NuaVIdK8cZ8ZOTi2M_Q?usp=sharing)***

  I have done the Assignment 5 by using the python package ***pytreebank*** to get the dataset. Below is the earlier submission link-

| Type | Link     |
| :-------- | :------- |
| `Colab` | ***[Prior Submission - Assignment 5 - colab](https://colab.research.google.com/drive/13-mpSe80XXG69Pz3y1SOP7HQmekAyggw?usp=sharing)*** |
| `Git` | ***[Prior Submission - Assignment 5 - Git Readme](https://github.com/SachinDangayach/END2.0/blob/main/Session5/README.md)*** |

**For this submission, we have prepared the dataset directly as explained in the steps below-**

  # Data

  1. We get the data files from the repo by link http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip. This is a popular dataset for benchmarking the sentiment analysis models. Sentences are broken into phrases and users have manually given the sentiment (5 classes from very negative to very positive) to each of these phrases which are the constituents of the sentences. we can see the breakdown tree structure of an example.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/SST_Tree.PNG)

  2. Once we get the data and unzip the files, we start preparing the dataframes for multiple files we have got.

  - Get sentiment values for phrases and convert label values to 5 buckets.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img1.PNG)

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img2.PNG)

  - Get all sentences in a dataframe.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img3.PNG)

  - Get the dictionary which contains all phrases and respective phrase ids.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img4.PNG)

  - In this dataset, each sentence is split into lowest level phrases/tokens and these phrases are provided the sentiments. These leaf level nodes are merged to get the sentiment of next level and this goes up till the entire sentence. Hence the dictionary contains all sentences with the sentiment as top level phrase or node. We can do an inner join to find these phrases.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img5.PNG)

  - As we need to get the sentiment of the sentences and sentiments are maintained at phrase level, we will do the inner join to get the sentiment of sentences using the dataframe from above step.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img6.PNG)

  - To get the train test split, we will not use the train_test split dataset which has got the sentence index and split label and instead we will split the data into 70:30 ratio directly

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img7.PNG)

  - Below is the final distribution of data amongst multiple classes

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img8.PNG)

  # The Network / Model - Architecture

  The network we decided to build is designed as follows:
  1. We have got embeddings layer with input dimension as vocab size (~15K) and embeddings dimension of 100.
  2. We used bidirectional RNN ( GRU) with 2 layers with input as embeddings from embeddings layer and hidden layer with 256 dimensions
  3. Dropout with (p= 0.3)
  4. Fully connected layer with input dimension as 512 (2 * hidden layer as its bidirectional RNN) and output of 5 dimension.

  Also, we have used the glove pre trained embeddings here by loading the pretrained embeddings weight to embeddings layer. We didn't free this layer at let the gradient to flow till this layer.

  The summary of the network looks like this:  

  ![Model Summary](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/img9.PNG)


  # The Network / Model - Loss Function

  We went ahead with the cross entropy loss with Adam optimizer with learning rate of 1e-5 for 80 epochs and then 1e-4 for next 20 epochs

  # The Training

  Trained the model for 100 epochs. Model started overfitting in most of the experiments.

  ### ACCURACY ACHIEVED - 40.46%
  Tried multiple experiment by couldn't get more than test 40% accuracy.

  ![Training logs](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/train_logs.PNG)

  Train Test Loss and Accuracy plots  

  ![Accuracy and Loss plots](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/loss_acc_plots.PNG)

  ### Model results on test dataset

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part1/tests.PNG)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2. Part  2: Train model we wrote in the class on the following two datasets taken from this links below:
  - http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.)
  - https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

## Solutions

| Type | Dataset     | Link     |
| :-------- | :------- | :------- |
| Part 2 A | Q & A Dataset | ***[Link for colab file](https://colab.research.google.com/drive/1CjHaQtXo2AVu4qzOG_UHGALelhUYI1mR?usp=sharing)*** |
| Part 2 B| Quora Dataset | ***[Link for colab file](https://colab.research.google.com/drive/1gVaUJifS7v8QNe0bO7NoEaKC05Ruv4lY?usp=sharing)*** |

### Part 2 A Data Preparation Steps

CMU Question-Answers dataset contains questions and answers from prepared from Wikipedia by three successive classes in CMU. We will download the dataset from the link and follow below mentioned steps -

- Download and unzip the dataset and load the files Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt, Question_Answer_Dataset_v1.2/S09/question_answer_pairs.txt , Question_Answer_Dataset_v1.2/S10/question_answer_pairs.txt into three separate dataframes.

  ![S08](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img1.PNG)

  ![S09](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img2.PNG)

  ![S10](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img3.PNG)

- Drop unnecessary columns to finally get dataframe with questions and answers as columns

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img4.PNG)

- Split the dataset containing ***3110*** records into train-test set with 70:30 split ratio.

- Create tokenizer to clean the dataset using spacy and create datafields [('Question', QUS), ('Answer', ANS)] with initial token as <sos>, end token as <eos>

- build the vocab using train data using tokens with min frequency = 2

- Create train and test iterators

### The Network / Model - Architecture

I have not done any changes to model as the intent was more on data prep. work.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img5.PNG)

### The Training
  Model achieves test ppl of 36.80 accuracy in 10 epochs
  - Training Log

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img6.PNG)


### Part 2 B Data Preparation Steps

An important product principle for Quora is that there should be a single question page for each logically distinct question. To mitigate the inefficiencies of having duplicate question pages at scale, we need an automated way of detecting if pairs of question text actually correspond to semantically equivalent queries. The dataset prepared to solve this challenging NLP problem consists of over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair.

We followed following steps to build a model which can generate a similar questions given a question as input.

- Download and unzip the dataset and load the files quora_duplicate_questions.tsv into dataframes. it contains ***404290*** records.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2B/img1.PNG)

- Filter records where question1 and question2 are duplicates. This results to ***149263*** records.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2B/img2.PNG)

- Drop unwanted columns and reset the index

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2B/img3.PNG)

- Split the dataset into train-test set with 70:30 split ratio.

- Create tokenizer to clean the dataset using spacy and create datafields =  [('question1', QUS1), ('question2', QUS2)] with initial token as <sos>, end token as <eos>

- build the vocab using train data using tokens with min frequency = 2

- Create train and test iterators

### The Network / Model - Architecture

I have not done any changes to model as the intent was more on data prep. work.

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img5.PNG)

### The Training
  Model achieves nearly test ppl of 35.7 accuracy in 10 epochs
  - Training Log

  ![alt](https://github.com/SachinDangayach/END2.0/blob/main/Session7/Utils/Part2A/img6.PNG)



3. Try for additional datasets the same activity what we tried above from the link mentioned below:
  - https://kili-technology.com/blog/chatbot-training-datasets
