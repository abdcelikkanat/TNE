# TNE
TNE: A Latent Model for Representation Learning on Networks

#### Installation
##### Anaconda installation
1.  Clone the repository by typing the following command:
```
git clone https://github.com/abdcelikkanat/TNE.git
```
2.  To initialize a new environment for Python 3.5 and to activate it, run the commands
```
conda create -n tne python=3.5
source activate tne
```
3. Install all the required modules.
```
pip3 install -r requirements.txt
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