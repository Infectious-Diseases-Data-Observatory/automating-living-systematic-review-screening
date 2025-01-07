ACRONYMS:

TIAB in the name of the models stands for Title and Abstract. These models use only the title and abstract of research articles for training and validation.

FT in the name of the models stands for Full text. These models use full text of research articles for training and validation.

Other acronyms:
 LR: Logistic Regression
SVM: Support Vector Machines
DNN: Deep Neural Network
LLM: Large Language Model



DATA:
 
Following models use data inside 'Title and abstract 10 fold data' folder: 
LR_TIAB, SVM_TIAB, DNN_TIAB*, DistilBERT_TIAB

Following models use data inside 'Full text 10 fold data' folder: 
LR_FT, SVM_FT, DNN_FT, DistilBERT_FT

LLM_Llama3.1_8b_instruct_zeroshot_TIAB uses Title_and_Abstract.csv

LLM_Llama3.1_8b_instruct_zeroshot_FT uses text.parquet

LLM_Llama3.1_8b_fineTune_TIAB uses data in 'Title and abstract data for fine tuning Llama' folder. It uses 5 fold cross validation.



*DNN_TIAB was the best performing model. 


