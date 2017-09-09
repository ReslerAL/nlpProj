# nlpProj
Train instruction:

1. Go to code/train.py and set the configuration you want to train
2. Run "python3 train.py"

Evaluation instruction:
1. If you want to evalute both the simple model and some lstm model run:
	"./eval.sh 0 1503663076748 0.3 false"
	where 1503663076748 is the model directory in saved/, 0.3 is the elimination ratio and fasle (or true) is verbose (the 0 argument is CUDA_VISIBLE_DEVICES variable for Nvidia GPU)

2. If you want to evaluate the simple model only run:
	python3 code/evaluate.py -model simple -embdsfile ./paragram-phrase-XXL.txt -evaldata ./evaluation_data.tsv [-verbose true] [-elimination 0.3]

3. If you want to evaluate the LSTM model only run:
	python3 code/evaluate.py -model lstm -dir /home/alon/workspace/nlp/saved/1503569471412/ -evaldata ./evaluation_data.tsv [-verbose true] [-elimination 0.3]
