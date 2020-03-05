# TNE
Topic-Aware Latent Models for Representation Learning on Networks

#### Installation
##### Anaconda installation
1.  Clone the repository by typing the following command:
```
git clone https://github.com/abdcelikkanat/TNE.git
```
2. To initialize a new environment for Python 3.6 and to activate it, run the commands
```
conda create -n tne python=3.6
source activate tne
```
3. Install all the required modules.
```
pip install -r requirements.txt
```


**Note:** _It may be required to compile the C extension of *gensim* package for a faster training process so you can run the following command:_
```
python setup.py install
```
_when you are inside the *"ext/gensim_wrapper/models/"* folder and copy the output **.so** file into this directory._

#### How to run
An example to learn node representations with *Louvain* community detection method might be
```
python run.py --corpus ./examples/corpus/karate.corpus --graph_path ./examples/datasets/karate.gml --emb ./karate.embedding --comm_method louvain

```
Similarly, we can adopt *LDA* algorithm in learning node representations.
```
python run.py --corpus ./examples/corpus/karate.corpus --emb ./karate.embedding --comm_method lda --K 2
```



You can view all the detailed list of commands by typing
```
python run.py -h
```

#### External Libraries
i) You might need to compile the source codes of **BigClam** and **GibbsLDA** algorithms for your operating system and place the executable files into suitable directories. You can also configure some parameters defined in the *consts.py* file.
