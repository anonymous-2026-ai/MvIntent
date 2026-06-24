# This is the implementation code of MvIntent: Multi-view representation learning for citation intent classification in scientific publications

## Environment setup 

1) Create python environment and install package in requirments.txt file.

2) Set up only single GPU for both training and testing by the command line: export CUDA_VISIBLE_DEVICES=0

## Multi-label classification: We train and test the multi-label classfication model on MultiCite dataset.

1. Training: Move to the folder: multi-label/train_multicite and run file: 'run.sh' 

2. Testing: First, you need to move the checkpoint you want to evaluate to the folder: multi-label/test_multicite, move to this folder and run file 'run.sh'

## Multi-class classification: We train and test the multi-class classfication model on ACL-ARC dataset.

1. Training: Move to the folder: multi-class/train_acl and run file: 'run.sh'

2. Testing: First, you need to move the checkpoint you want to evaluate to the folder: multi-class/test_acl, move to this folder and run file 'run.sh' 
