ResoNet: Robust and Explainable ENSO Forecasts with Hybrid
Convolution and Transformer Networks
Pumeng Lyu1, Tao Tang2, Fenghua Ling3, Jing-Jia Luo3, Niklas Boers4, Wanli
Ouyang1, and Lei Bai1,*
1Shanghai Artificial Intelligence Laboratory, Shanghai, 200072, China
2Zhejiang Meteorological Observatory, Hangzhou, 310002, China
3Institute for Climate and Application Research (ICAR), Nanjing University of
Information Science and Technology, Nanjing 210044, China
4Technical University of Munich, Potsdam Institute for Climate Impact Research
*Corresponding to bailei@pjlab.org.cn
Abstract
Recent studies have shown that deep learning (DL) models can skillfully predict the
El Nin˜o-Southern Oscillation (ENSO) forecasts over 1.5 years ahead. However, concerns
regarding the reliability of predictions made by DL methods persist, including potential
overfitting issues and lack of interpretability. Here, we propose ResoNet, a DL model that
combines convolutional neural network (CNN) and Transformer architectures. This hybrid
architecturedesignenablesourmodeltoadequatelycapturelocalSSTAaswellaslong-range
inter-basin interactions across oceans. We show that ResoNet can robustly predict ESNO
at lead times between 19 and 26 months, thus outerforming existing approaches in terms of
forecast horizon. According to an explainability method applied to ResoNet predictions of
El Nin˜o and La Nin˜a events from 1- to 18-month lead, we find that it predicts the Nin˜o3.4
index based on multiple physically reasonable mechanisms, such as the Recharge Oscillator
concept, Seasonal Footprint Mechanism, and Indian Ocean capacitor effect. Moreover, we
demonstratethatforthefirsttimetheasymmetrybetweenElNin˜oandLaNin˜adevelopment
can be captured by ResoNet. Our results could help to alleviate skepticism about applying
DL models for ENSO prediction and encourage more attempts to discover and predict
climate phenomena using AI methods.
Introduction
TheElNin˜o-SouthernOscillation(ENSO),characterizedbyirregularoscillationsbetweenwarm
(El Nin˜o) and cold (La Nin˜a) phases, is one of the most pronounced inter-annual climate vari-
abilitymodes,exertinginfluenceoverglobalclimatevariations[1]. Ithasattractedgreatinterest
since the 1980s [2, 3, 4]. Improvements in observing systems and ENSO prediction models help
current statistical or dynamical models effectively predict El Nin˜o events with a notable lead
time (i.e., 6 to 12 months) [5, 6, 7, 8]. However, slow oscillating signals in ENSO, such as
oceanic variations [5, 9], equatorial winds [10], and sea surface temperature anomalies (SSTA)
1
3202
ceD
61
]hp-oeg.scisyhp[
1v92401.2132:viXra

outside the equatorial Pacific [11], suggest that there remains untapped potential to extend
ENSO predictability to multiple years.
Deep learning methods have demonstrated remarkable advancements in various domains
over the past decade [12]. Inspired by the success of Convolutional Neural Networks (CNN) in
computervision[13,14],Hametal. (2019)usedathree-layerCNNmodelandachievedeffective
ENSO forecasts 17 months ahead [15]. Since then, various kinds of deep learning models have
beenadoptedinENSOforecasts[16,17,18,19,20]. Currently, twomajorkindsofdeeplearning
architectures are CNN and Transformers. CNN can efficiently handle datasets at different
scales but has limitations in modeling long-range interactions [21]. Transformers, based on a
self-attention-based architecture, can improve learning long-range interactions [21] and have
already shown enhanced performances in weather forecasts [22, 23]. Nevertheless, compared
with CNNs, Transformers lack locality and translation equivariance [21]. Therefore, a larger
numberoftrainingsamplesisrequiredtoeffectivelytrainapureTransformer. However, climate
dataatamonthlyscaleismuchsmallerthanweatherdataatanhourlyscale. Evenifwecanuse
historicalsimulationdatafortraining[15],climatedataisstillinsufficientcomparedwithimages
incomputervision(usuallyover100million)[21].OptmizingDLmodelswithinsufficienttraining
samples is likely to cause overfitting issues, which hinders the application of Transformer-only
models to capture climate dynamics like ENSO.
To address these challenges, we propose the Robust and Explainable ENSO forecasting
Network (ResoNet). ResoNet is an integrated network that combines CNNs and Transformers.
CombiningCNNsandTransformerscanimprovemodelperformancessincethishybridapproach
can adequately process local and global dynamics together [21, 24]. In addition, the bagging
algorithm [25] is applied to train 20 models with different parameter initializations and training
sets. With only three months’ SSTA as model input, ResoNet on average can make effective
predictions of Nin˜o3.4 index 18 months ahead in three test datasets we examined. We also
emphasized the potential risks of overfitting issues. Under ideal settings, ResoNet can make
effective forecasts up to 26 months in advance. However, this outstanding performance requires
the over-ideal selection of model hyperparameters.
Compared with the traditional dynamical or statistical methods, deep learning models for
ENSO predictions are often doubted due to their lack of physical explanations. In this work, we
2

employ the Integrated Gradient (IG) method [26] to explain the signals excavated by ResoNet
to provide a physical intepretation. ResoNet can effectively capture the Recharge Oscillator
paradigm[27,28],theSeasonalFootprintingMechanism[29],theinter-basininteractionsamong
tropicaloceans[30],andtheasymmetryofphasetransitionsbetweenElNin˜oandLaNin˜aevents
[31,32]. Theseoutcomesemphasizethemodel’scapacitytocapturemultipleintricatedynamics
within the climate system, providing valuable insights into the underlying mechanisms driving
| ENSO variation | and | predictability. |     |     |     |     |     |     |     |     |     |
| -------------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Results
| ResoNet | Achieves | Long-lead |     | ENSO |     | Predictions |     |     |     |     |     |
| ------- | -------- | --------- | --- | ---- | --- | ----------- | --- | --- | --- | --- | --- |
In this section, we present the quantitative prediction performance of ResoNet. Temporal
Anomaly Correlation Coefficient (ACC) and Root Mean Square Error (RMSE) are used to
validate the model performance. They are computed as a function of the forecast lead month t:
|     |        |     | 12       |           | (cid:80)e |        | −O¯         |           | −F¯ |     |     |
| --- | ------ | --- | -------- | --------- | --------- | ------ | ----------- | --------- | --- | --- | --- |
|     |        |     | (cid:88) |           |           | (O y,m | m )(F       | y,m,t     | m,t | )   |     |
|     | ACC(t) |     | =        |           | y=s       |        |             |           |     |     |     |
|     |        |     |          | (cid:113) |           |        |             |           |     |     | (1) |
|     |        |     |          | (cid:80)e |           | −O¯    | )2(cid:80)e |           | −F¯ |     |     |
|     |        |     | m=1      |           | (O        |        |             | (F        |     | )2  |     |
|     |        |     |          |           | y=s       | y,m    | m           | y=s y,m,t |     | m,t |     |
(cid:118)
|     |     |     |         |     | 12         | (cid:117)   | N        |     |     |     |     |
| --- | --- | --- | ------- | --- | ---------- | ----------- | -------- | --- | --- | --- | --- |
|     |     |     |         |     | 1 (cid:88) | (cid:117) 1 | (cid:88) |     |     |     |     |
|     |     |     | RMSE(t) | =   |            | (cid:116)   | (F       | −O  | )2  |     | (2) |
|     |     |     |         |     |            |             | y,m,t    |     | y,m |     |     |
|     |     |     |         |     | 12         | N           |          |     |     |     |     |
|     |     |     |         |     | m=1        |             | y=1      |     |     |     |     |
O¯ andF¯
Here,OandFdenotetheobservedandforecastvalues,respectively. denotethe
|     |     |     |     |     |     |     |     |     |     | m m,t |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
temporal climatology concerning target season m (from 1 to 12) and the forecast lead months
t (from 1 to 26). The label y denotes the forecast target year, respectively.
To align with Ham et al.(2019) [15], 32 years (1984-2017) SSTA from Extended Recon-
structed Sea Surface Temperature, version 5 from the National Oceanic and Atmospheric Ad-
ministration (ERSST.v5) were employed to examine and compare model performances. The
all-season correlation skills for the three-month-running-averaged Nin˜o3.4 index from 1984 to
2017 of different models are shown in Figure. 1a. Here, ResoNet-B represents the ensemble
mean predictions made from the 20 trained models using the bagging algorithm (see Methods).
3

Figure1: PerformancesofResoNetandotherpreviousdynamicalanddeeplearningmodels. (a)
All-season ACC of ResoNet, CNN, SINTEX-F, and eight NMME models at different forecast
lead months. (b) ACC of ResoNet using the bagging algorithm (ResoNet-B) at various lead
times in three validation datasets: ERSST.v5, ERA5, OISST.v2. (c) Same as (b) but RMSE
as performance metrics.
4

With the bagging algorithm, ResoNet models were trained 20 times separately using 20 differ-
ent subsets of CMIP6 data, and transfer learning was then applied to 100 years of SSTA on
ERSST.v5 (1871-1973). Compared with the CNN and dynamical models, ResoNet (ResoNet-B
and ResoNet-I) shows substrantially improved prediction skill based on correlations, especially
for long-term predictions with forecast lead beyond 11 months. The model exhibits all-season
correlation skills above 0.5 for forecast lead up to 19 months. To examine whether ResoNet has
overfitting problems, i.e., whether it is sensitive to different prediction test datasets or not, the
correlation skills and root mean squared error based on two other datasets (see Methods) are
also tested, i.e., ECMWF Reanalysis v5 (ERA5) [33] and NOAA Optimum Interpolation SST
v2 (OISST.v2). ACC of ResoNet consistently exceeds 0.5 around 18 months in advance in all
three datasets (see Fig. 1b), and the RMSE for the three different datasets (see Fig. 1c) does
not vary greatly, suggesting that our model is reliable and robust in long-term ENSO forecasts.
One advantage of using the bagging algorithm is that ensemble predictions do not rely on
validation data. We would like to highlight that AI models might achieve exaggerated perfor-
mance owing to their superior capability if trained with inappropriate settings, e.g., using the
testing set in the hyperparameter selection stage. For example, if sorting 20 trained models
and determining the ensemble of models from these 20 trained models based on the best per-
formances on testing data, ResoNet could even achieve an effective forecast lead of 26 months
ahead (see ResoNet-I in Fig. 1a and Supplementary Fig. 1). This results mean that our model
can make reliable, effective 18 months forecast while could potentially make skillful predictions
up to 26 months. However, testing data is always not available in real-time forecasts. Using
testing data influences ensemble model decisions and leads to overly optimistic performance
estimates since it can not generalize to new and unseen data.
ResoNet Captures Explainable El Nin˜o Evolution Dynamics
In addition to effective forecast skills, the physical mechanisms learned by ResoNet for El Nin˜o
predictions can be unraveled. Correlations between input SSTA and the predicted Nin˜o3.4
indexbyResoNetwerecomputedatdifferentforecastleadmonths(seeSupplementaryFig. S3).
Such linear regression method presents an overview of the linear dynamics ResoNet follows. To
furtherexploreregionsthatResoNetidentifiesasresponsibleresponsibleforskillfullypredicting
5

El Nin˜o, an explainable deep learning method called Integrated Gradient (IG) was applied [26].
Attribution values were calculated by Equation 3.
IntegratedGrad (x) = (x −x ′ )× (cid:88) m ∂F(x′ + m k(x−x′)) × 1 (3)
i i i ∂x m
i
k=1
Here, m is the number of steps in the Riemann approximation of the integral. i is the index
of the input pixels to the model, which in this paper, i goes from 1 to n = 3 × 24 × 72 =
5,184. x′ is the baseline we chose to compare outcomes. In this paper, the baseline is set as
the zero-embedding vector with the same shape as the input to the model (3 × 24 × 72). For
other details of IG, please refer to Sundararajan et al. (2017). Specifically, IntegratedGrad (x)
i
is the computed attribution value. The magnitude of attribution values can demonstrate which
regions are sensitive and greatly affect the predictive skill of El Nin˜o index. In this paper, 800
attribution maps were computed for the 20 trained models for 40 years (1982-2021) of inputs in
the DJF season. A sensitive region is defined where its attribution value during El Nin˜o years
surpasses the 95% confidence level, determined by comparing them to the averaged attribution
values from 1982 to 2021. In this paper, ten El Nin˜o years (1983, 1987, 1988, 1992, 1995, 1998,
2003, 2007, 2010, 2016) between 1982 and 2021 were selected to analyze sensitive regions during
El Nin˜o years.
TheadvantageofusingIGisthatitcancompareattributionvaluesduringElNin˜oyearsand
normal years. Composite input glkobal SSTA at different forecast leads to target DJF season
wereplotted(Fig. 2a-f), withsignificantregionsover95%confidencelevelbasedontheStudent
t-test shaded. Sensitive regions explored using IG on ResoNet align with significant regions in
general, which demonstrates that ResoNet effectively captures sensitive regions that are known
to play crucial roles in the formation of El Nin˜o events (see Fig. 2). When the forecast lead is
within a year, attribution values are significantly located in the tropical Pacific Ocean (see Fig.
2g-j). However, as the forecast lead increases, ResoNet pays more attention to signals outside
theequatorialPacific(seeFig. 2k, l), suchassignalsinthetropicalIndianOcean, NorthPacific
Ocean, and South Pacific Ocean. This switching attention of local and global information is
attributed to our model architecture, which combines CNN and Transformer.
FollowingtheRechargeOscillationmechanism[27],anElNin˜oeventhasbeenseededaccord-
ing to the equatorward Sverdrup transport of subsurface warm water during phase transition
6

Figure 2: Composite input SSTA and corresponding IG heatmaps in DJF season of 10 El
Nin˜o events (1983, 1987, 1988, 1992, 1995, 1998, 2003, 2007, 2010, 2016), according to NOAA
(https://www.psl.noaa.gov/enso/). (a)-(f) denotes the input SSTA with forecast lead at 3,
6, 9, 12, 15, and 18 months, respectively. Values over 95% confidence level based on Student’s
t-test are shaded. (g)-(l) denotes the attribution maps using the IG method with forecast
lead at 3, 6, 9, 12, 15, and 18 months, respectively. All 20 trained models are considered
for the attribution analysis. To avoid outliers that disrupt distributions of attribution values,
attributionmapswerefirstprocessedthroughaGaussianfilterwithsigma1.0. Onlyattribution
values that surpass the 95% confidence level are plotted.
7

from the cold to the warm phase [34]. Cold SSTAs over the central-eastern equatorial Pacific
keep the westerly wind anomaly, which further contributes to the poleward transport of surface
cold water by Ekman transport. More specifically, at a lead of 18 months, negative corre-
lations in the central and eastern tropical Pacific Ocean, along with positive correlations in
the off-equatorial tropical and eastern Pacific Ocean (see Supplementary Fig. 3f), indicate the
recharging process from the off-equatorial Pacific. This mechanism is well learned by ResoNet,
as suggested by the significant attribution values over the off-equatorial Pacific. More evident
signals appear over the southern Pacific than over the northern Pacific, which can be attributed
to the attenuating effect of temperature anomalies in the northern Pacific [35].
The equatorward transport of heat content leads to the deepening of the thermocline along
the equator. Accordingly, the upwelling of the warm water over the central-eastern tropical
Pacific gradually diminishes the cold SSTA there. The induced decrease of the trade wind over
the tropical western Pacific would further result in the easterly expansion of the warm pool (see
Fig. 2d, e). Correspondingly, the significant attribution values over the central Pacific imply
the process above has been learned accurately by ResoNet (see Fig. 2j). When the forecast lead
is within eight months, clear ENSO signals in the warm phase become evident (see Fig. 2a-c),
the SSTA over the central-eastern tropical Pacific could grow up persistently according to the
Bjerknes feedback [36]. Once again, the significant attribution values over the equator Pacific
suggest that the key region of El Nin˜o development and the associated mechanism have been
well captured by ResoNet (see Fig. 2g-i).
Another key region captured by ResoNet is located in the northeastern Pacific with a one-
year forecast lead (see Fig. 2j). The warm SSTA along the western coast of Northern America
can propagate to the central equatorial Pacific and generate an El Nin˜o events around boreal
spring according to the seasonal footprinting mechanism [29]. Additionally, the tropical Indian
Ocean is also considered by ResoNet as a key region with significant attribution values around
15 months ahead (see Fig. 2k). According to the Matsuno-Gill response [37, 38], negative
SSTA in the Indian Ocean induces divergence and westerly winds over the western tropical
Pacific Ocean. Consequently, the induced eastward propagation of downwelling Kelvin waves
potentially leads to the dissipation of La Nin˜a and initiation of El Nin˜o events, which is like the
mechanism suggested by Xie et al. (2009)[30]. Results here demonstrate that ResoNet catches
8

regional evolution dynamics in the tropical Pacific, but also long-range relationships outside the
tropical Pacific, thanks to the hybrid model architecture combining CNNs and Transformers.
ResoNet Explores Asymmetric Behaviors between El Nin˜o and La Nin˜a
One potential advantage of deep learning networks is their ability to build nonlinear and asym-
metric relations. In addition to the analysis of the formation of El Nin˜o events, we also used the
IG method to explore sensitive regions for La Nin˜a events at different forecast leads. Similar to
the analysis of El Nin˜o events, seven La Nin˜a years (1989, 1999, 2000, 2008, 2011, 2012, 2021)
were analyzed. Compared with El Nin˜o development, sensitive regions discovered during La
Nin˜a development (see Fig. 3) are somewhat different from El Nin˜o development, especially
when the forecast lead is 15-18 months, i.e., summer season a year before peak season. Results
hereindicatethatapartfromthelinearmechanismsmentionedabove, ResoNetcouldalsocatch
the asymmetry between El Nin˜o and La Nin˜a development well.
The physics behind the asymmetry between El Nin˜o and La Nin˜a development has been
widely studied [31, 32]. However, this has never been demonstrated in existing works about
applying AI methods in ENSO forecasts. More specifically, at the forecast lead of 18 months,
while ResoNet suggests that the development of El Nin˜o is influenced by the southern Pacific
(seeFig. 2l), itconsidersthatLaNin˜aeventsaremoresensitivetotheeasternequatorialPacific
(see Fig. 3l).
Based on the observational dataset (i.e., ERSST.v5), the asymmetry between the develop-
ment of El Nin˜o and La Nin˜a can be demonstrated in more detail (see Fig. 4). Here, 40 years
of Nin˜o 3.4 index in DJF season versus averaged SSTA in South Pacific and Nin˜o 3 region with
forecast lead of 18 months are plotted. Linear regressions of 40 years of Nin˜o 3.4 index are
shown in dashed green lines (see Fig. 3b and Fig. 3c). At forecast lead of 18 months, both
SSTA in South Pacific and East tropical Pacific have a negative correlation to Nin˜o 3.4 index
18 months later in general. However, considering El Nin˜o and La Nin˜a years other than normal
years, such correlations are somewhat asymmetric.
With only 10 El Nin˜o and 7 La Nin˜a years between 1982 and 2021, we are mainly concerned
with the location of quadrants for El Nin˜o or La Nin˜a years in scatter plots. Regarding the
SSTA over the southern Pacific, almost all El Nin˜o events are located in the second quadrant,
9

Figure 3: Composite input SSTA and corresponding IG heatmaps in DJF season of 7 La
Nin˜a events selected (1989, 1999, 2000, 2008, 2011, 2012, 2021), according to NOAA (https:
//www.psl.noaa.gov/enso/). (a)-(f) denotes the input SSTA with forecast lead at 3, 6, 9, 12,
15, and 18 months, respectively. Values over 95% confidence level based on Student’s t-test are
shaded. (g)-(l) denotes the attribution maps using the IG method with forecast lead at 3, 6, 9,
12, 15, and 18 months, respectively. All 20 trained models are considered for the attribution
analysis. To avoid outliers that disrupt distributions of attribution values, attribution maps
were first processed through a Gaussian filter with sigma 1.0. Only attribution values that
surpass the 95% confidence level are plotted.
10

which implies the SSTAs over the southern Pacific in the previous summer have qualitatively
consistent negative impacts on the El Nin˜o occur 18 months later. However, half of La Nin˜a
eventsareevenlylocatedinthethirdandfourthquadrants. Therefore,SSTAsoverthesouthern
Pacific have no qualitatively consistent impacts on the La Nin˜a events. The development of
La Nin˜a is significantly correlated with the SSTAs over the Nin˜o 3 region 18 months ahead.
In contrast, El Nin˜o events get no impacts from the SSTA averaged over Nin˜o3 region at 18
months ahead. This might be due to the slower demise of La Nin˜a than El Nin˜o as suggested
by previous work [39]. El Nin˜o events have stronger amplitude but decay quickly, so SSTA at
the tropical Pacific is more crucial for ESNO prediction during transitions from the warm phase
to the cold phase in summer. Therefore, La Nin˜a events are predictable regarding signals at
the tropical Pacific in the previous summer. However, stronger La Nin˜a events tend to last 2-3
years, so SSTA in tropical Pacific is not significant since it might complete transitions from cold
to the warm phase or cold SSTA persist and another La Nin˜a event appears in the following
year. The causes of this asymmetric evolution are due to the asymmetric SSTA pattern and
nonlinear atmosphere responses. Therefore, different sensitive regions between El Nin˜o and La
Nin˜a phase transitions suggest that ResoNet has correctly focused different regions according
to different phase transitions, demonstrating its ability in nonlinear and asymmetric modeling.
Discussion
Recently, deep learning methods have been widely applied to climate forecasts to push the
limits of prediction accuracy. Due to the lack of physical mechanisms, researchers might be
concerned about whether such good performances with AI methods are reliable. By designing
thenovelCNNandTransformerhybridmodelResoNetandtrainingwiththebaggingalgorithm,
our study here could avoid overfitting problems and enhance the performance, robustness, and
interpretabilityoftheResoNetmodel. ResoNetcouldskillfullyforecastENSO18monthsahead
reasonably due to the excellent capturing of the recharge oscillator mechanism, the inter-basin
interaction among tropical oceans, and the footprinting mechanism. In addition to improved
prediction skills compared to existing approaches, there should be more analysis of the hidden
dynamics AI models learn. For example, in this paper, our results demonstrate that ResoNet
can forecast El Nin˜o and La Nin˜a events based on different key regions due to very good
11

Figure 4: Correlations between the SSTA over the southern Pacific, Nin˜o3 region at 18 months
ahead and target Nin˜o3.4 index. (a) Locations of the southern Pacific (yellow, 25°- 10°S, 190°-
220 °E), and Nin˜o3 region (blue, 5°S - 5°N, 150°- 90 °W). (b) Scatter plot of averaged SSTA
over the southern Pacific versus target Nin˜o3.4 index. (c) Same as (b) but for the SSTA over
the tropical East Pacific Ocean (i.e., Nin˜o3 region).
12

capturing of the asymmetry between the development of El Nin˜o and La Nin˜a events. This
ability has never been achieved in previous AI models for ENSO forecasts. The development of
explainable deep learning methods thus encourages discoveries of hidden physical mechanisms
learned by deep learning models. While this study uses SSTA for model training, more complex
implementations, including multiplevariables, nonlinearinteractions, and advanced explainable
deep learning methods in climate forecasts, will be examined in future works. We hope our
resultscanhelpalleviatetheskepticismaboutapplyingArtificialIntelligencemethodsforENSO
predictionandsupportdecision-makingprocessesinvarioussectorsthatrelyonaccurateENSO
predictions.
Methods
Data
Historical simulations produced by 20 models from Coupled Model Intercomparison Project
phase 6 (CMIP6) were adopted to train ResoNet (see Supplementary Table S1). Only one
member was selected for each CMIP6 model. Therefore, a total of 3,240 monthly samples from
CMIP6 were used for training ResoNet, considering each target season and lead month. After
trainingResoNetwithCMIP6data,transferlearningwasconductedwith103yearsofreanalysis
data (1871-1973) from the Extended Reconstructed Sea Surface Temperature, version 5 from
the National Oceanic and Atmospheric Administration (ERSST.v5). Sea surface temperature
datain42years(1980-2021)fromNOAAExtendedReconstructionSSTsVersion5(ERSST.v5),
ECMWF Reanalysis v5 (ERA5), and NOAA Optimum Interpolation SST V2 (OISST.v2) were
downloaded to validate the performance of ResoNet forecast. All data mentioned was interpo-
lated into the regular grid (55°S - 60°N, 0°- 360 °E) with resolution 5°× 5°in both zonal and
meridional direction. Predictions from 8 models of the North American Multi-model Ensemble
(NMME) [40] at 1-11 months lead and SINTEX-F [41] at 1-23 months lead were collected to
compare ResoNet model performance with previous dynamical models.
To process sea surface temperature data for model training and inference, original gridded
data was first interpolated into the regular grid (55°S - 60°N, 0°- 360 °E) with resolution 5°×
5°in both zonal and meridional directions. Next, SSTA and the Nin˜o3.4 index were computed.
Finally, SSTA was normalized by the spatially averaged standard deviation for easier model
13

optimization. The number of forecast lead months is defined as the number of months between
the latest input data and the middle month of the target season. The three-month-running
mean of the Nin˜o3.4 index was the target output of ResoNet.
Architecture of ResoNet
ResoNet (see Fig. 5) uses direct forecast strategy [20] for each target season and forecast lead.
ResoNetconsistsofthreeparts: oneembeddinglayertoprocessthree-monthSSTAinputs,three
stagelayerstoextractlocalandglobalpatterns, andoneoutputlayertomakepredictions. Two
main components in ResoNet are the Cross-scale Embedding Layer (CEL) and ENSO Mobile
Block (EMB) in each stage layer. CELs in ResoNet are CNN-based layers that extract local
features. It uses two different 2-dimensional convolutional kernels (2×2 and 4×4). Therefore,
features at different scales and interactions between them can be extracted [42]. To reduce
model inference cost and speed up feature extraction, the stride of each convolutional kernel in
the CEL is set to 2×2, resulting in a reduction of the spatial dimension by a factor of 4.
ENSO Mobile Block (EMB) in ResoNet incorporates a self-attention-based Transformer be-
tween convolutional layers to explore global patterns [24]. Vision Transformer cuts images into
patches and uses self-attention layers to process these patches as tokens. However, EMB uses
Token Learner and Token Fuser to reduce model complexity and explore global relations effec-
tively [43]. Token Learner processed 2-dimensional input features with shape H ×W into S
tokens (see Supplementary Fig. 1a). Then, Transformer is applied to extract interactions be-
tweentheseStokens. TokenFuserthusprojectstheseprocessedStokensintoshapeH×W (see
Supplementary Fig. 1b). A few convolutional layers are used to process local information and
maintain data shape. By utilizing Token Learner and Token Fuser, EMB in ResoNet generates
tokensforTransformerwithouttheneedtodivideinputdataintonon-overlappingpatches. This
patch-free design ensures that no information will lost when cutting non-overlapping patches.
Thishybridandpatch-freedesignoftheENSOMobileBlockimprovestheeffectivenessofmodel
training and enables the capturing of crucial global features for accurate ENSO forecasts. The
detailed configuration of ResoNet is shown in Supplementary Table 2.
14

Figure 5: Overall model architecture of ResoNet. ResoNet takes SSTA for three consecutive
months as input. Global input SSTAs are embedded, concatenated, and mixed. Each stage
layerconsistsofaCross-ScaleEmbeddingLayer(CEL),whichisaCNN-basedarchitecturethat
learns spatial local features and their interactions at different scales. Each CEL is followed by
one ENSO mobile block (EMB), which is a Transformer-based architecture that captures global
patterns that are crucial for ENSO forecast. After three stage layers, the output layer makes
a three-month running mean of the Nin˜o3.4 index forecast. The right side of this figure shows
detailed illustrations of CEL and EMB. Skip connections are used so processed information
from both CNNs and Transformers can be kept for subsequent blocks.
15

| Bagging | algorithm |     |     |     |     |
| ------- | --------- | --- | --- | --- | --- |
Because of a small number of training samples and variations of ENSO data, the bagging
algorithm [25] was applied to reduce the instability of training and avoid overfitting. For each
target season and lead month, 20 different training sets, each of size 3,240, were generated by
sampling uniformly from 3,240 CMIP6 samples with replacement. Because every training set
was sampled with replacement, they were independent of each other and each is expected to
have the fraction (1 - 1/e)(≈ 63.2%) of 3,240 full CMIP6 samples (≈ 2,050 samples). The rest
CMIP6 samples not selected were used as validation sets for early stopping during training.
For each forecast lead month and target season, 20 different training sets from CMIP6 samples
weregeneratedtotrain20differentResoNetmodelswiththesamemodelstructurebutdifferent
modelweights. Transferlearningwasthenappliedtoall20modelswithERSST.v5(1871-1973).
Predictions were made by computing the ensemble mean of these 20 models.
| Model | training strategies |     |     |     |     |
| ----- | ------------------- | --- | --- | --- | --- |
ResoNet uses direct prediction strategy and predicts Nin˜o3. index. AdamW optimizer was used
for training [44]. To avoid gradient explosion and overfitting on training data, Smooth L1 Loss
was used as the loss function for backward propagation [45]. Equation 4 gives the computation
of Smooth L1 loss, with x and y denote the predictions and targets. Here, β was set to be
n n
0.5 for training ResoNet. Stochastic Gradient Descent with Restarts (SGDR) [46] learning rate
scheduler was used. The mini-batch size is 50 (20) for training on CMIP6 (ERSSTv5) samples.
The learning rate is set to 5.0e-5. Every training process stops when there is no improvement
of Smooth L1 Loss on the validation set for 5 epochs. The maximum number of epochs for
training with CMIP6 (ERSST.v5) samples is 100 (15). Detailed model training strategies are
| presented | in Supplementary | Table 3. |     |     |     |
| --------- | ---------------- | -------- | --- | --- | --- |

|     |     |  0.5(xn −yn)2 |     |              |       |
| --- | --- | -------------- | --- | ------------ | ----- |
|     |     |              | ,   | if |x n −y n | | < β |
β
l = (4)
n

|     |     |  |x −y | |−0.5·β, | otherwise |     |
| --- | --- | -------- | -------- | --------- | --- |
|     |     | n        | n        |           |     |
16

Data availability
| Data related    | to this paper                             | can be downloaded | from: |
| --------------- | ----------------------------------------- | ----------------- | ----- |
| CMIP6 database, | https://esgf-node.llnl.gov/search/cmip6/; |                   |       |
ERSST.v5 database, https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/;
ERA5 database, https://cds.climate.copernicus.eu/cdsapp#!/home;
OISST.v2 database, https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html,
| NMME, http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/ |     |     |     |
| ----------------------------------------------------------- | --- | --- | --- |
| Code availability                                           |     |     |     |
The deep learning models were developed using standard libraries in open-source platforms
including PyTorch (https://pytorch.org/). Codes used in this study are available from the
| corresponding | author | on request. |     |
| ------------- | ------ | ----------- | --- |
References
[1] James R Holton and Renata Dmowska. El Nin˜o, La Nin˜a, and the southern oscillation.
| Academic | press, | 1989. |     |
| -------- | ------ | ----- | --- |
[2] JP McCreary. A linear stratified ocean model of the equatorial undercurrent. Philosophical
TransactionsoftheRoyalSocietyofLondon.SeriesA,MathematicalandPhysicalSciences,
| 298(1444):603–635, |     | 1981. |     |
| ------------------ | --- | ----- | --- |
[3] SGH Philander, WJ Hurlin, and AD Seigel. Simulation of the seasonal cycle of the tropical
pacific ocean. Journal of Physical Oceanography, 17(11):1986–2002, 1987.
[4] JDNeelin, MojibLatif, MAFAllaart, MACane, UCubasch, WLGates, PRGent, MGhil,
CGordon,NCLau,etal. Tropicalair-seainteractioningeneralcirculationmodels. Climate
| Dynamics, | 7:73–104, | 1992. |     |
| --------- | --------- | ----- | --- |
[5] Jing-Jia Luo, Sebastien Masson, Swadhin K Behera, and Toshio Yamagata. Extended enso
predictions using a fully coupled ocean–atmosphere model. Journal of Climate, 21(1):84–
93, 2008.
17

[6] Anthony G Barnston, Michael K Tippett, Michelle L L’Heureux, Shuhua Li, and David G
DeWitt. Skillofreal-timeseasonalensomodelpredictionsduring2002–11: Isourcapability
increasing? Bulletin of the American Meteorological Society, 93(5):631–651, 2012.
[7] Hong-Li Ren, Adam A Scaife, Nick Dunstone, Ben Tian, Ying Liu, Sarah Ineson, June-Yi
Lee,DougSmith,ChangzhengLiu,VikkiThompson,etal. Seasonalpredictabilityofwinter
enso types in operational dynamical model predictions. Climate Dynamics, 52:3869–3890,
2019.
[8] Xinyang Wang, Joanna Slawinska, and Dimitrios Giannakis. Extended-range statistical
enso prediction through operator-theoretic techniques for nonlinear dynamics. Scientific
reports, 10(1):2636, 2020.
[9] ChuanGaoandRong-HuaZhang. Therolesofatmosphericwindandentrainedwatertem-
perature (t e) in the second-year cooling of the 2010–12 la nin˜a event. Climate Dynamics,
48:597–617, 2017.
[10] Geoffrey Gebbie and Eli Tziperman. Predictability of sst-modulated westerly wind bursts.
Journal of climate, 22(14):3894–3909, 2009.
[11] Jae-Heung Park, Jong-Seong Kug, Tim Li, and Swadhin K Behera. Predicting el nin˜o be-
yond1-yearlead: effectofthewesternhemispherewarmpool. Scientificreports,8(1):14957,
2018.
[12] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–
444, 2015.
[13] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep
convolutional neural networks. Advances in neural information processing systems, 25,
2012.
[14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for
image recognition. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016.
[15] Yoo-Geun Ham, Jeong-Hwan Kim, and Jing-Jia Luo. Deep learning for multi-year enso
forecasts. Nature, 573(7775):568–572, 2019.
18

[16] Paul Johannes Petersik and Henk A Dijkstra. Probabilistic forecasting of el nin˜o using
neural network models. Geophysical Research Letters, 47(6):e2019GL086423, 2020.
[17] Salva Ru¨hling Cachay, Emma Erickson, Arthur Fender C Bucker, Ernest Pokropek, Willa
Potosnak, Salomey Osei, and Bj¨orn Lu¨tjens. Graph neural networks for improved el ni\˜
no forecasting. arXiv preprint arXiv:2012.01598, 2020.
[18] Jining Yan, Lin Mu, Lizhe Wang, Rajiv Ranjan, and Albert Y Zomaya. Temporal convo-
lutional networks for the advance prediction of enso. Scientific reports, 10(1):8055, 2020.
[19] Mayuna Gupta, Hariprasad Kodamana, and SJIG Sandeep. Prediction of enso beyond
spring predictability barrier using deep convolutional lstm networks. IEEE Geoscience and
Remote Sensing Letters, 19:1–5, 2020.
[20] Bin Mu, Bo Qin, and Shijin Yuan. Enso-gtc: Enso deep learning forecast model with a
global spatial-temporal teleconnection coupler. Journal of Advances in Modeling Earth
Systems, 14(12):e2022MS003132, 2022.
[21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
ThomasUnterthiner, MostafaDehghani, MatthiasMinderer, GeorgHeigold, SylvainGelly,
et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv
preprint arXiv:2010.11929, 2020.
[22] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Pangu-
weather: A 3d high-resolution model for fast and accurate global weather forecast. arXiv
preprint arXiv:2211.02556, 2022.
[23] Kang Chen, Tao Han, Junchao Gong, Lei Bai, Fenghua Ling, Jing-Jia Luo, Xi Chen,
Leiming Ma, Tianning Zhang, Rui Su, et al. Fengwu: Pushing the skillful global medium-
range weather forecast beyond 10 days lead. arXiv preprint arXiv:2304.02948, 2023.
[24] Sachin Mehta and Mohammad Rastegari. Mobilevit: light-weight, general-purpose, and
mobile-friendly vision transformer. arXiv preprint arXiv:2110.02178, 2021.
[25] Leo Breiman. Bagging predictors. Machine learning, 24:123–140, 1996.
19

[26] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep net-
works. In International conference on machine learning, pages 3319–3328. PMLR, 2017.
[27] Fei-Fei Jin. An equatorial ocean recharge paradigm for enso. part i: Conceptual model.
| Journal | of the | atmospheric | sciences, | 54(7):811–829, |     | 1997. |
| ------- | ------ | ----------- | --------- | -------------- | --- | ----- |
[28] Tianming Li. Phase transition of the el nin˜o–southern oscillation: A stationary sst mode.
| Journal | of the | atmospheric | sciences, | 54(24):2872–2887, |     | 1997. |
| ------- | ------ | ----------- | --------- | ----------------- | --- | ----- |
[29] Daniel J Vimont, John M Wallace, and David S Battisti. The seasonal footprinting mech-
anism in the pacific: Implications for enso. Journal of climate, 16(16):2668–2675, 2003.
[30] Shang-Ping Xie, Kaiming Hu, Jan Hafner, Hiroki Tokinaga, Yan Du, Gang Huang, and
Takeaki Sampe. Indian ocean capacitor effect on indo–western pacific climate during the
| summer | following | el nin˜o. | Journal | of climate, | 22(3):730–747, | 2009. |
| ------ | --------- | --------- | ------- | ----------- | -------------- | ----- |
[31] Gerrit Burgers and David B Stephenson. The “normality” of el nin˜o. Geophysical Research
| Letters, | 26(8):1027–1030, |     | 1999. |     |     |     |
| -------- | ---------------- | --- | ----- | --- | --- | --- |
[32] Soon-Il An and Fei-Fei Jin. Nonlinearity and asymmetry of enso. Journal of Climate,
| 17(12):2399–2412, |     | 2004. |     |     |     |     |
| ----------------- | --- | ----- | --- | --- | --- | --- |
[33] HansHersbach,BillBell,PaulBerrisford,ShojiHirahara,Andr´asHor´anyi,Joaqu´ınMun˜oz-
Sabater, Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, et al. The era5
global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730):1999–
2049, 2020.
[34] Wenjun Zhang, Sixu Li, Fei-Fei Jin, Ruihuang Xie, Chao Liu, Malte F Stuecker, and
Aoyun Xue. Enso regime changes responsible for decadal phase relationship variations
between enso sea surface temperature and warm water volume. Geophysical Research
| Letters, | 46(13):7546–7553, |     | 2019. |     |     |     |
| -------- | ----------------- | --- | ----- | --- | --- | --- |
[35] Niklas Schneider, Arthur J Miller, Michael A Alexander, and Clara Deser. Subduction
of decadal north pacific temperature anomalies: Observations and dynamics. Journal of
| Physical | Oceanography, |     | 29(5):1056–1070, | 1999. |     |     |
| -------- | ------------- | --- | ---------------- | ----- | --- | --- |
20

[36] Jakob Bjerknes. Atmospheric teleconnections from the equatorial pacific. Monthly weather
| review, | 97(3):163–172, | 1969. |     |     |     |     |
| ------- | -------------- | ----- | --- | --- | --- | --- |
[37] Taroh Matsuno. Quasi-geostrophic motions in the equatorial area. Journal of the Meteo-
| rological | Society | of Japan. Ser. | II, | 44(1):25–43, | 1966. |     |
| --------- | ------- | -------------- | --- | ------------ | ----- | --- |
[38] Adrian E Gill. Some simple solutions for heat-induced tropical circulation. Quarterly
| Journal | of the | Royal Meteorological |     | Society, | 106(449):447–462, | 1980. |
| ------- | ------ | -------------------- | --- | -------- | ----------------- | ----- |
[39] Mingcheng Chen, Tim Li, Xinyong Shen, and Bo Wu. Relative roles of dynamic and
thermodynamic processes in causing evolution asymmetry between el nin˜o and la nin˜a.
| Journal | of Climate, | 29(6):2201–2220, |     | 2016. |     |     |
| ------- | ----------- | ---------------- | --- | ----- | --- | --- |
[40] Ben P Kirtman, Dughong Min, Johnna M Infanti, James L Kinter, Daniel A Paolino,
Qin Zhang, Huug Van Den Dool, Suranjana Saha, Malaquias Pena Mendez, Emily Becker,
et al. The north american multimodel ensemble: phase-1 seasonal-to-interannual predic-
tion; phase-2 toward developing intraseasonal prediction. Bulletin of the American Mete-
| orological | Society, | 95(4):585–601, | 2014. |     |     |     |
| ---------- | -------- | -------------- | ----- | --- | --- | --- |
[41] Jing-Jia Luo, Sebastien Masson, Erich Roeckner, Gurvan Madec, and Toshio Yamagata.
Reducing climatology bias in an ocean–atmosphere cgcm with improved coupling physics.
| Journal | of climate, | 18(13):2344–2360, |     | 2005. |     |     |
| ------- | ----------- | ----------------- | --- | ----- | --- | --- |
[42] W Wang, L Yao, L Chen, B Lin, D Cai, X He, and W Liu. Crossformer: A versatile vision
transformer hinging on cross-scale attention. arxiv 2021. arXiv preprint arXiv:2108.00154,
2018.
[43] MichaelSRyoo, AJPiergiovanni, AnuragArnab, MostafaDehghani, andAneliaAngelova.
Tokenlearner: What can 8 learned tokens do for images and videos? arXiv preprint
| arXiv:2106.11297, |     | 2021. |     |     |     |     |
| ----------------- | --- | ----- | --- | --- | --- | --- |
[44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint
| arXiv:1711.05101, |     | 2017. |     |     |     |     |
| ----------------- | --- | ----- | --- | --- | --- | --- |
[45] RossGirshick. Fastr-cnn. InProceedings of the IEEE international conference on computer
| vision, | pages 1440–1448, | 2015. |     |     |     |     |
| ------- | ---------------- | ----- | --- | --- | --- | --- |
21

[46] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts.
| arXiv | preprint | arXiv:1608.03983, |     | 2016. |
| ----- | -------- | ----------------- | --- | ----- |
Acknowledgements
This work is supported by Shanghai Artificial Intelligence Laboratory and National Foundation
| of China | (Grant        | 42030605). |     |     |
| -------- | ------------- | ---------- | --- | --- |
| Author   | contributions |            |     |     |
P.M.L. and T.T. are co-first authors. L.B. is the corresponding author who conceived the idea
of this study. P.M.L and L.B. designed the AI models. P.M.L performed experiments. P.M.L.
and T.T. performed the analysis and wrote the manuscript under the supervision of J.-J.L.,
N.B., L.B., and W.O.. All authors contributed to interpreting results, discussions of associated
| dynamics,   | and improvement |              | of the     | presentation. |
| ----------- | --------------- | ------------ | ---------- | ------------- |
| Competing   |                 | interests    |            |               |
| All authors | declare         | no competing | interests. |               |
22

| Supplementary |        | Information   |     |       |          |          |        |     |      |              |             |
| ------------- | ------ | ------------- | --- | ----- | -------- | -------- | ------ | --- | ---- | ------------ | ----------- |
|               |        | Supplementary |     | Table |          | 1: CMIP6 | models |     | used | for training |             |
|               | Source | ID            |     |       | Modeling |          | Groups |     |      |              | Integration |
Period
| ACCESS-ESM-1-5 |         |     |     | Commonwealth |          | Scientific |              | and       | Industrial |     |     |
| -------------- | ------- | --- | --- | ------------ | -------- | ---------- | ------------ | --------- | ---------- | --- | --- |
|                |         |     |     |              | Research |            | Organisation |           |            |     |     |
|                | CanESM5 |     |     | Canadian     | Centre   | for        | Climate      | Modelling |            | and |     |
Analysis
|     | CESM2         |     |        | National | Center   | for           | Atmospheric |                   | Research |     |     |
| --- | ------------- | --- | ------ | -------- | -------- | ------------- | ----------- | ----------------- | -------- | --- | --- |
|     | CESM2-FV2     |     |        | National | Center   | for           | Atmospheric |                   | Research |     |     |
|     | CESM2-WACCM   |     |        | National | Center   | for           | Atmospheric |                   | Research |     |     |
|     | CNRM6-CM6-1   |     | Centre | National |          | de Recherches |             | M´et´eorologiques |          |     |     |
|     | EC-Earth3-Veg |     |        |          | EC-Earth |               | consortium  |                   |          |     |     |
|     | FGOALS-f3-L   |     |        |          | Chinese  | Academy       |             | of Sciences       |          |     |     |
1850-2014
|                 | GISS-E2-1-G  |     |     | NASA       | Goddard | Institute       |        | for Space       |     | Studies |     |
| --------------- | ------------ | --- | --- | ---------- | ------- | --------------- | ------ | --------------- | --- | ------- | --- |
| HadGEM3-GC31-LL |              |     |     |            | Met     | Office          | Hadley | Centre          |     |         |     |
|                 | ICON-ESM-LR  |     |     | Max        | Planck  | Institute       |        | for Meteorology |     |         |     |
|                 | IPSL-CM6A-LR |     |     | Institute  |         | Pierre          | Simon  | Laplace         |     |         |     |
|                 | MCM-UA-1-0   |     |     | Department |         | of Geosciences, |        | University      |     | of      |     |
Arizona
|     | MIROC6 |     | Japan | Agency |     | for Marine-Earth |     |     | Science | and |     |
| --- | ------ | --- | ----- | ------ | --- | ---------------- | --- | --- | ------- | --- | --- |
Technology
| MPI-ESM-1-2-HAM |               |     |         | Max            | Planck     | Institute |             | for Meteorology |           |     |     |
| --------------- | ------------- | --- | ------- | -------------- | ---------- | --------- | ----------- | --------------- | --------- | --- | --- |
|                 | MPI-ESM1-2-HR |     |         | Max            | Planck     | Institute |             | for Meteorology |           |     |     |
|                 | MPI-ESM1-2-LR |     |         | Max            | Planck     | Institute |             | for Meteorology |           |     |     |
|                 | MRI-ESM2-0    |     |         | Meteorological |            |           | Research    |                 | Institute |     |     |
|                 | NESM3         |     | Nanjing |                | University | of        | Information |                 | Science   | and |     |
Technology
|     | UKESM1-0-LL |     |                | National |     | Institute      | of  | Meteorological |     |     |     |
| --- | ----------- | --- | -------------- | -------- | --- | -------------- | --- | -------------- | --- | --- | --- |
|     |             |     | Sciences/Korea |          |     | Meteorological |     | Administration |     |     |     |
23

|              | Supplementary |     | Table     | 2:      | Details of configurations |     |            | of ResoNet. |     |          |
| ------------ | ------------- | --- | --------- | ------- | ------------------------- | --- | ---------- | ----------- | --- | -------- |
| Module       |               |     |           | Layer   |                           |     | Resolution |             |     | Channels |
| Input        |               |     |           | -       |                           |     | 24         | × 72        |     | 1        |
|              |               |     | Conv3×3   |         |                           |     | 24         | × 72        |     | 1 → 16   |
| 2d Embedding |               |     | GroupNorm |         |                           |     | 24         | × 72        |     | 16       |
|              |               |     | LeakyReLU |         |                           |     | 24         | × 72        |     | 16       |
| 1 st CEL     |               |     | Conv2×2,  | Conv4×4 |                           | 24  | × 72       | → 12 ×      |     | 16 → 96  |
36
|          |     | Conv3×3, |             | GroupNorm, | SiLU     |     | 12   | × 36    |     | 96     |
| -------- | --- | -------- | ----------- | ---------- | -------- | --- | ---- | ------- | --- | ------ |
|          |     | Conv1×1, |             | GroupNorm, | SiLU     |     | 12   | × 36    |     | 96     |
|          |     |          | Token       | Learner    |          |     | 12 × | 36 → 2  |     | 3 × 96 |
| 1 st EMB |     |          | Transformer |            | Layer ×3 |     |      | 2       |     | 3 × 96 |
|          |     |          | Token       | Fuser      |          |     | 2 →  | 12 × 36 |     | 96     |
|          |     | Conv1×1, |             | GroupNorm, | SiLU     |     | 12   | × 36    |     | 96     |
|          |     | Conv3×3, |             | GroupNorm, | SiLU     |     | 12   | × 36    |     | 96     |
| 2 nd CEL |     |          | Conv2×2,    | Conv4×4    |          | 12  | × 36 | → 6 ×   |     | 96     |
18
|     |     | Conv3×3, |       | GroupNorm, | SiLU |     | 6   | × 18   |     | 96     |
| --- | --- | -------- | ----- | ---------- | ---- | --- | --- | ------ | --- | ------ |
|     |     | Conv1×1, |       | GroupNorm, | SiLU |     | 6   | × 18   |     | 96     |
|     |     |          | Token | Learner    |      |     | 6 × | 18 → 4 |     | 3 × 96 |
nd
| 2 EMB |     |          | Transformer |            | Layer ×3 |     |     | 4      |     | 3 × 96 |
| ----- | --- | -------- | ----------- | ---------- | -------- | --- | --- | ------ | --- | ------ |
|       |     |          | Token       | Fuser      |          |     | 4 → | 6 × 18 |     | 96     |
|       |     | Conv1×1, |             | GroupNorm, | SiLU     |     | 6   | × 18   |     | 96     |
|       |     | Conv3×3, |             | GroupNorm, | SiLU     |     | 6   | × 18   |     | 96     |
rd
| 3 CEL |     |          | Conv2×2, | Conv4×4    |      | 6   | × 18 | → 3 × 9 |     | 96     |
| ----- | --- | -------- | -------- | ---------- | ---- | --- | ---- | ------- | --- | ------ |
|       |     | Conv3×3, |          | GroupNorm, | SiLU |     | 3    | × 9     |     | 96     |
|       |     | Conv1×1, |          | GroupNorm, | SiLU |     | 3    | × 9     |     | 96     |
|       |     |          | Token    | Learner    |      |     | 3 ×  | 9 → 4   |     | 3 × 96 |
rd
| 3 EMB  |       |          | Transformer     |               | Layer ×3 |     |     | 4       |     | 3 × 96     |
| ------ | ----- | -------- | --------------- | ------------- | -------- | --- | --- | ------- | --- | ---------- |
|        |       |          | Token           | Fuser         |          |     | 4 → | 3 × 9   |     | 96         |
|        |       | Conv1×1, |                 | GroupNorm,    | SiLU     |     | 3   | × 9     |     | 96         |
|        |       | Conv3×3, |                 | GroupNorm,    | SiLU     |     | 3   | × 9     |     | 96         |
|        |       | Global   |                 | Average       | Pooling  | 3   | × 9 | → 1 × 1 |     | 3 × 96     |
|        |       |          | Layer           | Normalization |          |     | 1   | × 1     |     | 3 × 96     |
| Output | Layer |          |                 |               |          |     |     |         |     |            |
|        |       |          |                 | Flatten       |          |     | 1   | × 1     | 3   | × 96 → 288 |
|        |       |          | Fully Connected |               | Layer    |     | 1   | × 1     |     | 288 → 1    |
24

|              | Supplementary | Table 3: ResoNet | model training | strategies.       |
| ------------ | ------------- | ---------------- | -------------- | ----------------- |
|              |               | CMIP6 Training   |                | Transfer Learning |
| Batch Size   |               | 50               |                | 20                |
| Epochs       |               | 100              |                | 15                |
| Optimizer    |               | AdamW            |                | AdamW             |
| Initial LR   |               | 5.0e-5           |                | 5.0e-5            |
| Scheduler    |               | SGDR             |                | SGDR              |
| Weight Decay |               | 0.0001           |                | 0.0001            |
25

(5)
Supplementary Figure 6: (a) Token Learner and (b) Token Fuser used in ENSO mobile block
(EMB). With MLP layer and matrix multiplication, Token Learner transfers information from
H ×W grid points into S tokens. After Transformer layers, Token Fuser takes S tokens and
input to Token Learner and projects feature shape back to H ×W grid points. Here, C is the
number of feature channels.
26

(6)
Supplementary Figure 7: Performances of ResoNet of 20 trained models and three ensemble
modeling methods at forecast lead months from 1 to 26. (a) Correlations of all-season three-
month averaged Nin˜o3.4 index at forecast lead from 1 to 26 months. Distributions of 20 single
models are displayed as green-shaded regions. (b) Same as (a) but for root mean square error
(RMSE).
27

(7)
Supplementary Figure 8: Correlation between ResoNet’s input and output at forecast lead from
1 to 18 months. Correlations are computed by linear regression of DJF season from 1982 to
2021. Significant regions (p ≤ 0.01) according to linear regression are shaded.
28

(8)
Supplementary Figure 9: Composite SSTA inputs of 10 El Nin˜o events at forecast lead from 1
to 18. Values over 95% confidence level based on Student’s t-test are shaded.
29

(9)
Supplementary Figure 10: Composite SSTA inputs of 7 La Nin˜a events at forecast lead from 1
to 18. Values over 95% confidence level based on Student’s t-test are shaded.
30

(10)
Supplementary Figure 11: Attribution maps of 10 El Nin˜o events in DJF Season at forecast
lead months from 1 to 18. All 20 trained models are considered for the attribution analysis.
Only attribution values that surpass the 95% confidence level, determined by comparing them
to the averaged attribution values from 1982 to 2021, are plotted.
31

(11)
Supplementary Figure 12: Attribution maps of 7 La Nin˜a events in DJF Season at forecast lead
months from 1 to 18. All 20 trained models are considered for the attribution analysis. Only
attribution values that surpass the 95% confidence level, determined by comparing them to the
averaged attribution values from 1982 to 2021, are plotted.
32