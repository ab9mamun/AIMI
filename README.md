# AIMI: Leveraging Future Knowledge and Personalization in Sparse Event Forecasting for Treatment Adherence
### _Abdullah Mamun, Diane J. Cook, Hassan Ghasemzadeh_

This repository contains the code and resources for **AIMI**, a knowledge-guided recurrent neural network system for forecasting medication adherence. 

## Overview  
We have conducted a user study with people who take medications for cardiovascular diseases and developed forecasting models for early detection of medication non-adherence.  Our proposed system for forecasting medication adherence is a method based on deep recurrent neural networks that uses sensor data and knowledge available about future events (e.g., time when a person is prescribed to take medication). We present detailed results on how much of the forecasting performance can be improved by using future knowledge. Moreover, we have designed the system to be compatible for incremental and personalized training for better performance making the system compatible for training in powerful computation nodes with GPUs as well as regular workstations with limited computation power.

## Citation 
If you would like to use part of our code or dataset or mention this work in your paper, please cite the following publication:

**_1. AIMI: Leveraging Future Knowledge and Personalization in Sparse Event Forecasting for Treatment Adherence_**
````
@misc{mamun2025aimileveragingfutureknowledge,
      title={AIMI: Leveraging Future Knowledge and Personalization in Sparse Event Forecasting for Treatment Adherence}, 
      author={Abdullah Mamun and Diane J. Cook and Hassan Ghasemzadeh},
      year={2025},
      eprint={2503.16091},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.16091}, 
}
````

## Key Features
- **Knowledge-Guided Forecasting:** AIMI incorporates prior knowledge about user behavior and medication history to enhance adherence predictions.
- **Sensor Integration:** Utilizes data from smartphone sensors, making it a convenient and widely accessible solution.
- **Incremental Training Pipeline:** The AIMI system is designed to work for both high-end GPU-based training on large servers and CPU-based training with regular workstations with low resources.
- **Personalized Support:** Enables the potential for on-demand intervention tools tailored to individual needs.

## User Study
A user study was conducted involving 27 participants who manage cardiovascular diseases with daily medications. The study collected data to evaluate the effectiveness of AIMI's forecasting models.

## Model Architecture
We developed and evaluated two types of forecasting models:
1. **Convolutional Neural Networks (CNNs):** Designed for feature extraction from sensor data.
2. **Long Short-Term Memory Networks (LSTMs):** Optimized for sequential data and time-series forecasting.
3. **Variations of LSTMs:** Different combinations of knowledge-guided, knowledge-free, context-rich, context-free, and location-based forecasting options.

### Performance and Findings
- **Accuracy:** 0.932 for forecasting next-hour medication adherence.
- **F1-Score:** 0.936 for forecasting next-hour medication adherence.
- **Statistical Significance:** The p-value for the effect of future knowledge on F-1 scores is 0.00006 << standard threshold of 0.05.

### Ablation Studies
Ablation studies demonstrated that:
- Incorporating known future knowledge enhances model performance.
- Personalized training improves adherence forecasting accuracy.

## How AIMI Works
1. **Data Collection:** Smartphone sensors gather activity data, while users log their medication intake.
2. **Feature Processing:** Sensor data and medication history are processed to create input features for the models.
3. **Adherence Prediction:** The AIMI forecaster forecasts whether a person is likely to miss their medication.


## Contact
For questions, suggestions, or bug reports: a.mamun@asu.edu
#### Read our other papers: https://abdullah-mamun.com
