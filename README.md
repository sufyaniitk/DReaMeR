# DReaMeR - Dynamic Readaptation and Memory Retention

### A CS771 Course Project

## Installation  

To install all the dependencies required for this project, use the `requirements.txt` file. Follow the steps below:  

1. Ensure you have Python 3.8 or above installed.  
2. Install `pip` if it is not already available.  
3. Run the following command in your terminal:  

```bash
pip install -r requirements.txt
```

* Ensure datasets are correctly loaded in the required structure before running the notebook.
* We *strongly recommend* the following directory structure
  ```
  - main
  | -- extracted_feature
  | -- dataset
  | -- venv (if using one)
  | -- LwP[MAIN].ipynb
  | -- ...
  ```
  where `...` of course represents the other `.py` files in this repo. Note the folder name `extracted_feature`


`LwP[MAIN].ipynb` is our main file. The file is sufficiently self-contained so far as the project is concerned. Ensure you have `dataset` and `extracted_feature` in the same directory as the `.ipynb`.

[Download `extracted_feature`](https://drive.google.com/drive/folders/1LvjHfk7grWC4PVFVwhHWYb_oAA0JBMwI?usp=drive_link)

## Overview

The project addresses countering unseen domains in Machine Learning models. Traditional classifiers tend to overfit the data they're "provided" due to sampling bias. For example, models that fail to identify newer strains of SARS-CoV-2 correctly. Even if the training set is diverse, the same model may fail on data drawn from the same dataset but with a different probability distribution.

This can be mitigated by selectively training the model on confident samples and even duplicating those samples with small random noise to increase the model's confidence in that locale. We attempt to implement this novel idea presented by Liu, et al. (2023) in [*Deja vu: Continual Model Generalization for Unseen Domains*](https://arxiv.org/abs/2301.10418)

Copyright Notice
© 2024 Adarsh Pal, Mohd Amir, Mohd Sufyan, Nishant Pandey. All Rights Reserved.

This repository and its contents are created solely for academic purposes as part of CS771 coursework at IIT Kanpur. Unauthorized reproduction or use of this code for commercial or unethical purposes is strictly prohibited.


Authors:
* Adarsh Pal | [adarshp22@iitk.ac.in](adarshp22@iitk.ac.in) (220054)
* Mohd Amir | [mmamir22@iitk.ac.in](mmamir22@iitk.ac.in) (220660)
* Mohd Sufyan | [msufyan22@iitk.ac.in](msufyan22@iitk.ac.in) (220662)
* Nishant Pandey | [nishantp22@iitk.ac.in](nishantp22@iitk.ac.in) (220724)
