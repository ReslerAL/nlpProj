# nlpProj

Train instruction. From the project directory run:

0. get the word embeddings for the simple model from Weiting and unzip it:
	-> wget http://ttic.uchicago.edu/~wieting/paragram-phrase-XXL.zip
	-> unzip paragram-phrase-XXL.zip -d .
	-> rm paragram-phrase-XXL.zip
1. Open code/train.py and set the training configuration you want at the top of the file
2. Run "python3 code/train.py"

Evaluation instruction:
1. If you want to evalute both the simple model and some lstm model run:
	"./eval.sh 0 1503663076748 0.3 false"
	where 1503663076748 is the model directory in saved/, 0.3 is the elimination ratio and fasle (or true) is verbose (the 0 argument is CUDA_VISIBLE_DEVICES variable for Nvidia GPU)

2. If you want to evaluate the simple model only run:
	python3 code/evaluate.py -model simple -embdsfile ./paragram-phrase-XXL.txt -evaldata ./eval_data_nlml_version2.tsv [-verbose true] [-elimination 0.3]

3. If you want to evaluate the LSTM model only run (change 1503569471412 to your model directory):
	python3 code/evaluate.py -model lstm -dir ./saved/1503569471412/ -evaldata ./eval_data_nlml_version2.tsv [-verbose true] [-elimination 0.3]
