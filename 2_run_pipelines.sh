#!/bin/bash
########################################################################################################################
# 2_run_pipelines.sh - Runs the train and predict for each dataset to train and then generate predictions
########################################################################################################################

########################################################################################################################
# Data - Is Epic Intro
########################################################################################################################
python train.py -d "path/to/data/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models/Is Epic/"
python predict.py -d "path/to/data/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/models/Is Epic/" -o "path/to/predictions/Is Epic Intro Full.csv"

########################################################################################################################
# Data - Needs Respray
########################################################################################################################
python train.py -d "path/to/data/Data - Needs Respray - 2024-03-26" -l "Labels-NeedsRespray-2024-03-26.csv" -t "Needs Respray" -o "path/to/models/Needs Respray/"
python predict.py -d "path/to/data/Data - Needs Respray Full" -l "Needs Respray Files.txt" -t "Needs Respray" -m "path/to/models/Needs Respray/" -o "path/to/predictions/Needs Respray Full.csv"
python verify.py "path/to/predictions/Needs Respray Full.csv" "path/to/data/Data - Needs Respray Full/Labels-NeedsRespray-2024-04-12.csv" "Needs Respray" "Needs Respray"

########################################################################################################################
# Data - Is GenAI
########################################################################################################################
python train.py -d "path/to/data/Data - Is GenAI - 2024-03-25" -l "Labels-IsGenAI-2024-03-25.csv" -t "Is GenAI" -o "path/to/models/Is GenAI/"
python predict.py -d "path/to/data/Data - Is GenAI Full" -l "Is GenAI Files.txt" -t "Is GenAI" -m "path/to/models/Is GenAI/" -o "path/to/predictions/Is GenAI Full.csv"
python verify.py "path/to/predictions/Is GenAI Full.csv" "path/to/data/Data - Is GenAI Full/Labels-IsGenAI-2024-04-12.csv" "Is GenAI" "IsGenAI"

########################################################################################################################
# Data - Tomato
########################################################################################################################
python train.py -d "path/to/data/Data - Tomato Leaf Disease" -l "Labels-IsDease-2024-04-16.csv" -t "Is Disease" -o "path/to/models/Is Disease/"
python predict.py -d "path/to/data/Data - Tomato Leaf Disease Full" -l "Is_Tomato_Leaf_Disease.txt" -t "Is Disease" -m "path/to/models/Is Disease/" -o "path/to/predictions/Is Disease Full.csv"
python verify.py "path/to/predictions/Is Disease Full.csv" "path/to/data/Data - Tomato Leaf Disease Full/Labels-IsDease-2024-04-12.csv" "Is Disease" "Is Disease"
