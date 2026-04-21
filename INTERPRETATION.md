# INTERPRETATION

## Data Insights

### Emergency Encounters
![alt text](assets/emg_enc_hist.png)
![alt text](assets/emg_enc_curr.png)
The number of emergency room encounters has gone down between the two periods.

### Patient Emergency Risk
![alt text](assets/risk_comp.png)
There are fewer high risk patients currently than historically.

### Age distribution
![alt text](assets/age_hist.png)
![alt text](assets/age_curr.png)
As expected, the historical dataset has an approximately normal distribution of ages centred around 60, significantly older than the current dataset.

### Emergency Encounters vs Income
![alt text](assets/inc_enc.png)
Lower income patients are more susceptible to emergency room visits.

### Healthcare Coverage
![alt text](assets/health_cov.png)
The distribution of healthcare coverage has become heavily skewed to lower coverage per patient between the two periods.

## Model Interpretation

### SVC
![alt text](assets/svc_hist.png)
![alt text](assets/svc_curr.png)
The historical model does about equally when scored on accuracy on both datasets. Finetuning does not appear to have a significant improvement, suggesting that the level of stationarity required by the SVC is met.

### Decision Tree
![alt text](assets/dtree_hist.png)
![alt text](assets/dtree_curr.png)
Finetuning on current data significantly improves the decision tree's metrics. Overall, the decision tree classification performs the best for the problem at hand.

### MLP
![alt text](assets/mlp_hist.png)
![alt text](assets/mlp_curr.png)
The MLP models perform the poorest for the problem at hand, however, finetuning does seem to help slightly. The issue could be the insufficient training.