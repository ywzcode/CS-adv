# Domain Adaptation with Cauchy-Schwarz Divergence
---
### Demo code for, Domain Adaptation with Cauchy-Schwarz Divergence, UAI2024. 

The core part of CS divergence is in CSutils.py.

In order to train the model, please download the Office-Home dataset from the official website. 

Then modify the data path in line 211 in train_image_officehome_cs.py. 

### Run 
sh run_office_home.sh  will train the Art to Clipart task by default. 


Our code is adapted from [CGDM](https://github.com/lijin118/CGDM) and [MCD_DA](https://github.com/mil-tokyo/MCD_DA).

