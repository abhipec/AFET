## AFET (baseline experiment)

Please follow the original instructions available at  https://github.com/shanzhenren/AFET for data pre-processing.

Sanitized directory contains the entity mention ids used in this experiment for comparison on same training, development and test set.

### Publication 

Fine-Grained Entity Type Classification by Jointly Learning Representations and Label Embeddings. Abhishek, Ashish Anand and Amit Awekar. EACL 2017. 

### baseline run
```bash
bash run_Wiki.sh
bash run_OntoNotes.sh
bash run_BBN.sh
```
*Note*
Please change python virtualenv path in the files mentioned above. 

### type-wise analysis
```bash
python3 class_wise_score.py dataset_name
```
