# TNE
Topic-Aware Latent Models for Representation Learning on Networks

#### Installation
##### Anaconda installation
1.  Clone the repository by typing the following command:
```
git clone https://github.com/abdcelikkanat/TNE.git
```
2.  To initialize a new environment for Python 3.6 and to activate it, run the commands
```
conda create -n tne python=3.6
source activate tne
```
3. Install all the required modules.
```
pip install -r requirements.txt
```


**Note:** _It may be required to compile "word2vec_inner.pyx" file._

#### How to run
```
python run.py --graph_path ./datasets/karate.gml --random_walk deepwalk --n 80 --l 10 --k 2 --community_detection_method lda --negative 5 --output_folder ./outputs --concat_method max
```

The detailed list of commands can be seen by typing
```
python run.py -h
```

#### Prerequisites
1.  You might need to compile the C extension of *gensim* package for a faster training process so you can run the following command 
```
python setup install
```
and copy the output **.so** file into the folder *"ext/gensim_wrapper/models/"*

2.  In order to use the **BigClam** algorithm, you should compile the sources and put the executable file in the following directory:
```
ext/agm-package/
```