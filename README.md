# Semantic Classification using Natural Language Processing
## By Ezeh Stanley

https://github.com/CodeyBank/Natural-Language

The algorithms for solving the above problem are implemented in Python (3.7 and above). Libraries used for the project include:
- NumPy
- Pandas
- Matplotlib
- re
- sklearn
- nltk
- mlxtend (1.14.0)

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

### Training a model and evaluating performance
```sh
python main.py <dataset_path> <command> <classifier_name(s)>
```

@args:
dataset_path - Absolute path of the document folder containing the dataset

command - Select what action to take
 - '-t' - test an already trained model
 - '-rs' - generate classifier model based on selected classifier algorithms by passing desired classifier names (below)

classifier_names - The classifier names are:

- KN - K-Neighbors 
- DT - Decision Tree 
- OR - One Rule
- SVC - SVM 
- NN - Neural Network 
- NB - Naive Bayes 
- RF - Random Forest

After the program is run, the trained model is saved in the project folder and a plot of the cross-validation scores for each
classifier is plotted and displayed

### Testing the model
To load the model use the following command for testing and evaluation, use the following command

NB: No plots will be made this time

```sh
python main.py <dataset_path> -t <model_path>  
```
@args
model_path- the absolute path to the model (saved in the project directory).
dataset_path -the absolute path to the documents to be analysed
