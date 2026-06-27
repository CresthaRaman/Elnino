Deep learning for multi-year ENSO forecasts
Yoo-Geun Ham(1*), Jeong-Hwan Kim(1) & Jing-Jia Luo(2,3)
10.29
1. Department of Oceanography, ChonnamNational University, Gwangju, South Korea.
2. Institute for Climate and Application Research (ICAR)/CICFEM/KLME/ILCEC, Nanjing University of Information Science and Technology, Nanjing, China.
3. SKLLQG, Institute of Earth Environment, Chinese Academy of Sciences, Xi’an, China.
*e-mail: ygham@jnu.ac.kr
https://doi.org/10.1038/s41586-019-1559-7

Outline
• ENSO forecast deadlock
• Deep learning setup
• Training strategy
• Forecast performance
• Further analysis
• Heat map analysis
• El Nino flavor forecast
• Conclusion

ENSO forecast deadlock
State-of-the-art dynamical forecast systems CANNOT make skillful prediction of ENSO for lead time longer than a year.
BUT:
• The presence of an oscillating element in ENSO, linked to slowly varying oceanic variations and their coupling to
the atmosphere, suggests that multi-year forecasts are possible.
• Equatorial Pacific anomalies during several La Niña events lingered for several years.
• The slowly varying component of the equatorial winds coupled with underlying SST is predictable to some extent.
• SST anomalies outside the equatorial Pacific can lead to an ENSO event with a time-lag longer than a year.
It is implied that there is still room for improvement in ENSO prediction!

Deep learning setup
|     | Nino3.4 index: area-averaged SST anomaly over 170°–120° |     |     | W, 5° S–5° | N   |
| --- | ------------------------------------------------------- | --- | --- | ---------- | --- |
•
| Input domain range: 0°–360° |     | E and 55° | S–60° N |     |     |
| --------------------------- | --- | --------- | ------- | --- | --- |
• Input data: SST and heat content (upper 300 m) anomaly
• Output is three-month-averaged Nino3.4 index (different model for different lead time)
•
4 different setups (M & N can both be 30 or 50) are applied to deploy ensemble forecast
• Zero-padding is applied to maintain the size during convolution process

Training strategy
Data shortage dilemma:
Observations of global oceanic temperature distributions are available from 1871. This means that, for each calendar month,
the number of samples is less than 150.
Transfer learning technique:
Train the model on similar task with large amount of data first. Then fine tune the model on the target task with limited data.
Standard training Fine tune
Initial model Trained model Final model
CMIP5 output Reanalysis data

Dataset details:
• A ten-year gap between the latest year in the training period and the earliest year in the validation period is
leaved to remove the possible influence of oceanic memory in the training period on the ENSO in the validation
period.
• The systematic errors in the CNN, reflecting those of the CMIP5 samples, are corrected after the second training
period using reanalysis.
• The year in the CMIP5 models is solely dependent on the prescribed greenhouse gas forcing, so no observational
information was added to the CMIP5 historical simulations.
• Take one ensemble member per model from all 21 models of CMIP5.

Forecast performance
The all-season correlation skill of the three-
month-moving-averaged Nino3.4 index. The
shading denotes 95% confidence interval.
The correlation skill of the Nino3.4 index
targeted to each calendar month in the
CNN model and the SINTEX-F dynamical
forecast system. Hatching highlights the
forecasts with correlation skill exceeding
0.5.

What we can see:
• The forecast skill of the Nino3.4 index in the CNN model is systematically superior to all state-of-the-art dynamical
prediction systems at lead times longer than six months.
• The CNN model is one of two models with the best forecasting skills for the first six forecast lead months.
• The all-season correlation skill of the Nino3.4 index in the CNN model is above 0.5 for a lead of up to 17 months,
whereas it is 0.37 at a lead of 17 months in the SINTEX-F.
• It is concluded that the CNN model provides a skillful forecast of ENSO events up to 1.5 years in advance: a result
that is not possible using any of the state-of-the-art forecast systems.
• The CNN model also shows a higher correlation skill of the Nino3.4 index for almost all targeted seasons, compared
to SINTEX-F
• The correlation skill improvement is especially robust for predictions targeting the seasons between the late boreal
spring and autumn.
• The CNN model is less affected by spring predictability barriers.
Deep learning wins!

Further analysis
The skill of the FC-NN model with a different number of predictors
FC-NN model setup
Input: EOF-PC of SST and heat contents in Indo-
Pacific, Atlantic and North Pacific regions
Output: Nino3.4 index
Training data: reanalysis data from 1871 to 1973
& CMIP5 model output
Model structure: 2 hidden layers with 20 nodes
Activation function: hyperbolic tangent function
Best performance: 9 PCs in Indian-Pacific region and 7 PCs in Atlantic, North Pacific region

Best FC-NN model vs CNN model
The ability to predict an ENSO in the CNN is
systematically superior to that in the neural
network model.

Transfer learning effect
No transfer learning experiment replaces
reanalysis data with corresponding CMIP5
output (the same period) to maintain the
same training data size.
Transfer learning technique contributes a
lot to the final performance.

Training dataset sensitivity
No significant reliance on
training dataset arrangement
CMIP5 model data difference:
CNN_org: the first ensemble member of all CMIP5 models
EXP1: the second ensemble member of all CMIP5 models
EXP2: randomly select a member if model provides 2 or more members
EXP3: randomly select 21 members out of all 42 members (total number same as former exps)
EXP4: the training samples are randomly selected for each year among 42 realizations

Validation dataset sensitivity
The similar correlation skill between
EXP1 and EXP3 indicates that the
forecast skill is not much changed
due to the different sampling for
the CNN model training.
The low correlation skill in EXP2
indicates that the CNN model has
difficulty when predicting the
modeled ENSO index.
EXP1 train: All CMIP5 output + reanalysis(1871-1973), vali: reanalysis(1984-2017)
EXP2 train: First half of all CMIP5 output, vali: last half of CMIP5 output
EXP3 train: Same as EXP2, vali: same as EXP1

Prediction on model ENSO index
The training samples are the first half of the historical
simulations in all CMIP5 models. While the validation
set is the last half of different models.
The standard deviation of the SST anomalies during DJF
season in reanalysis, well-predicted models and poor-
predicted models during 1979 to 2001.
This shows that the correlation skill of the CNN model is
quite stable and robust even predicting the modeled
ENSO as far as the ENSO dynamics is simulated realistically.
The validation dataset in unrealistic models is responsible
for the low correlation skill in EXP2 to a large extent.

Heat map analysis
The Nino3.4 index for the DJF season for the 18-month-
lead forecast demonstrates that the CNN model correctly
predicts the ENSO amplitude. But why?
The heat map of the predictors for the DJF season in
1997/98. Positive (negative) values in the heat maps
denote predictors over certain regions contributing to
the prediction of a positive (negative) Nino3.4. The
anomalies over the tropical western Pacific, Indian
Ocean and subtropical Atlantic are the main contributors
to the successful prediction of the 1997/98 El Nino.

El Nino flavor forecast
Gray zone indicates 95% confidence interval. The
CNN model overcomes a long-standing weakness
of the state-of-the-art forecast models
Another CNN model to predict El Nino flavor. The CNN output corresponds
to the percentage occurrence of three El Nino categories: CP-type, EP-type
and a mixture of the two. Only the CMIP5 model outputs is used as
training data.

Heat map analysis
EP-type CP-type
Area-averaged heat map value of EP-type and CP-type over
south Pacific, equatorial Pacific, north Pacific, Indian Ocean, and The CNN can be a powerful tool to reveal complex ENSO
equatorial Atlantic. First two cases having the biggest heat map mechanisms!
value for each ocean basin are selected as the most favorable
pattern for EP or CP development.

Conclusions
• The CNN model provides a skillful forecast of ENSO events up to 1.5 years in advance:
a result that is not possible using any of the state-of-the-art forecast systems.
• The skill uncertainty produced by changing the training and validation dataset is
small, indicating that the CNN can provide skillful real-time forecasts.
• The CNN model is even successful in predicting the modelled ENSO index in some of
the CMIP5 models that capture realistic ENSO dynamics.
• The CNN model can predict the spatial complexity of the El Nino events with great
precision by skillfully predict ENSO flavor.
• The CNN model can be a powerful tool to reveal complex ENSO mechanisms.