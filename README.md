# Semantic Classification using Natural Language Processing
## By Ezeh Stanley

https://github.com/CodeyBank/Natural-Language

The algorithms for solving the above problem are implemented in Python (3.7 and above). Libraries used for the project include:
- [NumPy]
- [Pandas]
- [Matplotlib]
- [re]
- [sklearn]
- [nltk]
- [mlxtend] (1.14.0)

Download and install these dependencies
```sh
pip install pandas
pip install mathplotlib
pip install numpy
pip install nltk
pip install mlxtend 
```

the Nltk (Natural Language Tool Kit) automatically downloads two dependencies 
- Wordnet
- Stopwords

## Steps to Run the program
The following steps are to be taken to run the program
-	Open Command Line Tool. In windows, this can be done from the Command Line Prompt. In Linux Based Operating systems, it can be accessed from Terminal.
-	Change directory to project directory
-	In the command line interface, type the following


```sh
python main.py <classifier_name>
```

The classifier names are:

- KN - K-Neighbors 
- DT - Decision Tree 
- OR - One Rule
- SVC - SVM 
- NN - Neural Network 
- NB - Naive Bayes 

After the program is run, the trained model is saved in the project folder.
To load the model use the following command for testing and evaluation, use the following command

```sh
python main.py test <model_path> <dataset_path> 
```

model_path is the math to the model (saved in the project directory).
dataset_path is the path to the documents to be analysed


To plot the results of the analysis for comparison, type the following into the command line
```sh
python main.py plot
```

### boxPlot is designed to work in environments where the classifier objects are saved after instantiation such as Spyder and Jupyter Notebook
