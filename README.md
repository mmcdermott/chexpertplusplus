# `Chexpert++`
## Description
Source implementation and pointer to pre-trained models for `chexpert++` (arxiv link forthcoming) a BERT-based
approximation to CheXpert for radiology report labeling. Note that a compelling, co-discovered alternative is
[1], which features a more full-fledged annotation effort featuring two board-certified radiologists and a
more robust error resolution system. This paper is accessible [here](://arxiv.org/pdf/2004.09167.pdf).

## Obtaining our Pre-trained Model
Our Pre-trained BERT model is soon to be available via PhysioNet. In the meantime, please email
[`mmd@mit.edu`](mailto:mmd@mit.edu) to request the pre-trained model, and be prepared to provide verification
that you have authorization to access MIMIC-CXR.

## Installation
To install a conda environment suitable for reproducing this work, use the environment spec available in
`env.yml`, via, e.g.
```
conda env create -f env.yml -n [ENVIRONMENT NAME]
```

Additionally, you must download the [MIMIC-CXR dataset](https://physionet.org/content/mimic-cxr/2.0.0/) and
split the reports into sentences, then label each of these with the CheXpert labeler (code/splits not
provided). You must also download the Clinical BERT model, available
[here](https://github.com/EmilyAlsentzer/clinicalBERT).

## Usage Instructions
Main model source code is available in `./chexpert_approximator`. Model training, evaluation, and active
learning proof-of-concept are all available in `Jupyter Notebooks/`.

## Citation
*This Work:*
Matthew B.A. McDermott, Tzu Ming Harry Hsu, Wei-Hung Weng, Marzyeh Ghassemi, and Peter Szolovits.
"`Chexpert++`: Approximating the CheXpert labeler for Speed, Differentiability, and Probabilistic Output."
Machine Learning for Health Care (2020) _(in press; link TBA)_.

*[1]*
Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Y. Ng, and Matthew P. Lungren. "CheXbert:
Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT." arXiv
preprint arXiv:2004.09167 (2020). [https://arxiv.org/pdf/2004.09167.pdf](https://arxiv.org/pdf/2004.09167.pdf)
