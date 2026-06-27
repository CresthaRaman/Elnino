npj | climate and atmospheric science
Article
PublishedinpartnershipwithCECCRatKingAbdulazizUniversity
https://doi.org/10.1038/s41612-026-01360-5
Enhancing the predictability limits of
ENSO with physics-guided deep echo
state networks
Checkforupdates
ZejingZhang1,JunMeng2 ,ZhongpuQiu3,WansuoDuan2,JianGao1,4,ZixiangYan1,JinghuaXiao1 ,
XiaosongChen3,WenjuCai5,6,7,8,JürgenKurths9,10,ShlomoHavlin11&JingfangFan3,9
TheElNiño-SouthernOscillation(ENSO)isadominantmodeofinterannualclimatevariability,yetthe
mechanismslimitingitslong-leadpredictabilityremainunclear.Here,wedevelopaphysics-guided
DeepEchoStateNetwork(DESN)thatoperatesonphysicallyinterpretableclimatemodesselected
fromtheextendedrechargeoscillator(XRO)framework.DESNachievesskillfulNiño3.4predictionsup
to16–20monthsaheadwithminimalcomputationalcost.Mechanisticexperimentsshowthat
extendedpredictabilityarisesfromnonlinearcouplingbetweenwarmwatervolumeandinter-basin
climatemodes.Error-growthanalysisfurtherindicatesafiniteENSOpredictabilityhorizonof
approximately30months.Theseresultsdemonstratethatphysics-guidedreservoircomputing
providesanefficientandinterpretableframeworkfordiagnosingandpredictingENSOatlong
leadtimes.
The El Niño-Southern Oscillation (ENSO) is the dominant mode of demonstratethatdynamicalcouplingwithinthetropicalPacificandwith
interannualclimatevariability,arisingfromnonlinearocean-atmosphere remote oceans provides robust precursors of ENSO evolution, enabling
interactions in the tropical Pacific and exerting far-reaching impacts on forecastsbeyondtheseasonaltimescale23,24.Theseinsightsareformalizedin
global climate extremes, ecosystems, agriculture, and socio-economic the Extended Recharge Oscillator (XRO), which elevates ENSO from a
systems1–6.ClassicalconceptualmodelsdistilltheessentialphysicsofENSO, basin-confinedoscillatortoamulti-basincoupleddynamicalsystemand
fromtheDelayedOscillator,whichemphasizesdelayednegativefeedbacks extendspredictiveskilltoroughly18months25.
associated with equatorial wave reflections7,8, to the Recharge Oscillator While XRO represents a major advance toward physically inter-
(RO),whichframesENSOasacoupledevolutionofseasurfacetemperature pretablelong-leadENSOprediction,existingconceptualmodelsarefor-
and equatorial subsurface heat content governed by Bjerknes feedback, mulatedinareducedstatespacewithprescribedfeedbacks,emphasizing
delayedoceanicadjustment,stochasticforcing,andnonlinearatmospheric physical interpretability and diagnostic clarity. In parallel, deep learning
processes9–13.State-of-the-artcoupledgeneralcirculationmodelsreproduce (DL)modelshavedemonstratedsubstantialpotentialforlong-rangeENSO
manyobservedENSOcharacteristicsandachieveskillfulforecastsupto prediction by flexibly extracting complex spatiotemporal patterns from
approximatelyoneyear14,15. high-dimensional climate data. Landmark studies26 demonstrated that
However, accumulating evidence suggests that ENSO dynamicsare purelydata-drivenconvolutionalnetworkscouldrivalphysicalmodels,and
embeddedwithinabroader,multiscaleinter-basinframeworkinvolvingthe recent transformer-based architectures such as 3D-Geoformer27 have
IndianOcean,theextratropicalPacific,andAtlanticvariability16–22.Data- extendedNiño3.4forecastskillbeyond18months.Despitetheirpredictive
drivenapproachesbasedonnetworktheoryandcomplexitysciencefurther success, these architectures are computationally demanding and often
1SchoolofPhysicalScienceandTechnology,BeijingUniversityofPostsandTelecommunications,Beijing,China.2StateKeyLaboratoryofEarthSystem
NumericalModelingandApplication,InstituteofAtmosphericPhysics,ChineseAcademyofSciences,Beijing,China.3SchoolofSystemsScience/Instituteof
NonequilibriumSystems,BeijingNormalUniversity,Beijing,China.4StateKeyLaboratoryofInformationPhotonicsandOpticalCommunications,BeijingUni-
versityofPostsandTelecommunications,Beijing,China.5FrontierScienceCenterforDeepOceanMultispheresandEarthSystem(FDOMES)andPhysical
OceanographyLaboratory,OceanUniversityofChina,Qingdao,China.6LaoshanLaboratory,Qingdao,China.7StateKeyLaboratoryofMarineEnvironmental
ScienceandCollegeofOceanandEarthSciences,XiamenUniversity,Xiamen,China.8StateKeyLaboratoryofLoessandQuaternaryGeology,InstituteofEarth
Environment,ChineseAcademyofSciences,Xi’an,China.9PotsdamInstituteforClimateImpactResearch,Potsdam,Germany.10InstituteofPhysics,Humboldt-
University,Berlin,Germany.11DepartmentofPhysics,Bar-IlanUniversity,Ramat-Gan,Israel. e-mail:jun.meng.phy@gmail.com;jhxiao@bupt.edu.cn;
jingfang@bnu.edu.cn
npjClimateandAtmosphericScience| ( 2026) 9:92 1
;,:)(0987654321 ;,:)(0987654321

https://doi.org/10.1038/s41612-026-01360-5 Article
difficulttointerpretphysically,whichcomplicatesmechanisticdiagnosis, DESNachieveshighskillandlong-termENSOforecasts
robustnessassessment,andregimetransfer. We trained our DESN models on the ORAS5 dataset over the period
These contrasting strengths raise a central challenge: can machine 1958–1999andevaluatedthemonindependentdatafrom2002–2023,
learning,whenguidedbyphysicallymotivatedconceptualframeworkssuch followinganout-of-sampleforecastingprotocol(SupplementaryAlgo-
as XRO, achieve strong long-lead predictive skill while also providing a rithms and Table S2). Forecast skill was assessed using the anomaly
physicallygroundedunderstandingofENSOdynamics?Here,“physically correlationcoefficient(ACC)oftheNiño3.4indexacrossarangeoflead
grounded understanding” refers to identifying the interaction pathways, times.Asbenchmarks,wecomparedDESNagainst(1)aminimalEcho
timescales, and nonlinear structures that are dynamically essential for StateNetwork(ESN),astandardsingle-layerreservoircomputingmodel
predictability,ratherthanestablishingstrictphysicalcausality. trainedonlyontheNiño3.4index,(2)afullESNtrainedontenclimate
Toaddressthischallenge,wedevelopaphysics-guidedandcompu- modes(Fig.1a)andseasonalcycles,(3)theXROmodel25,(4)operational
tationallyefficientforecastingframeworkbasedonDeepEchoStateNet- ensembleforecastsfromtheInternationalResearchInstituteforClimate
works (DESNs), a class of reservoir computing models well suited for and Society (IRI), and (5) the state-of-the-art 3D-Geoformer deep
nonlinear dynamical systems28–31. Unlike conventional deep learning learning model27. While all models performed similarly at short lead
architectures,DESNcombineshierarchicalreservoirdynamicswithphy- times,DESNshowedclearadvantagesbeyond10months,maintaining
sically interpretable inputs, enabling both long-lead prediction and sys- ACC values above 0.5 up to 16 months (Fig. 2a). It consistently out-
tematicdynamicaldiagnosis.Themodelincorporatesseasonalpriorsand performedtheIRIdynamicalensemblebeyond9monthsandsurpassed
ten ENSO-related climate indices selected by the XRO framework25, thestatisticalensembleafterjust6months.
allowingnonlinearcross-basininteractionstoberepresentedwithoutpre- To evaluate seasonal sensitivity, we computed target-month-
scribingexplicitcouplingstructures. dependentforecastskillusingrollinginitializations.TheminimalESN
TheproposedDESNframeworkachievesNiño3.4forecastanomaly maintainedACC>0.5onlyatshortleadsacrossallmonths(Fig.2c).The
correlation coefficients exceeding 0.5 at lead times of 16–20 months, fullESNextendedthishorizontoabout15monthsbetweenJanuaryand
comparabletostate-of-the-artphysicalanddeeplearningmodels25–27, June(Fig.2d),butitsperformancedroppedsharplyaroundJuneand
while requiring only seconds of training on a standard CPU. Beyond July, reflecting the Spring Predictability Barrier (SPB)40. In contrast,
predictive skill, DESN supports mechanistic diagnosis through sub- DESNpreservedelevatedskillthroughthisseason.Forexample,fore-
model and controlled experiments, revealing that nonlinear, state- casts targeting June achieved a 10-month skillful lead with DESN
dependentcouplings-particularlythoseinvolvingwarmwatervolume comparedtoonly6monthsfortheESN(Fig.2e).Theseresultsindicate
and cross-basin interactions-play a central role in sustaining predict- that extending the predictor set from the minimal to the full ESN
abilitybeyondcanonicalrecharge-dischargedynamics.Buildingonthe highlightstheimportanceofcross-basincouplingasasourceofENSO
error-growth framework underlying nonlinear local Lyapunov expo- predictability, whereasthe furtherimprovementfromthe full ESN to
nent (NLLE) analyses32,33, we examine the saturation behavior of DESN arises from the deep architecture’s ability to represent higher-
forecast errors using an absolute-error metric. The resulting error order,state-dependentcouplingsandmultiscaleinteractionsthatpro-
growthdynamicsindicateanintrinsicENSOpredictabilityhorizonof videadditionalpredictiveinformation.
approximately30months. Wefurtherassessedmodelgeneralizationoverextendedperiods,
Rather than prioritizing forecast accuracy alone, this work demon- applyingthesamehyperparameters(TableS2)asintheout-of-sample
stratesthatincorporatingphysicallyinterpretablestructureintolightweight implementationtoensureconsistentdynamicalbehavior.Evaluation
recurrentmodelsenablesinterpretableandtheory-consistentENSOpre- was performed over two temporal periods: 1979 to 2023 for the in-
diction.ByguidingaDeepEchoStateNetworkwithclimatemodesmoti- sampleexperiment(Fig.2b)and1958to2023forcross-validation(see
vatedbytheXROframework,weshowhownonlinearcouplingstructures “Methods”).Inthein-sampletest,theminimalESNrapidlylostskill
andtheintrinsicpredictabilitylimitsimposedbyENSOdynamicsjointly after 3 months, likely reflecting overfitting. The full ESN sustained
shapeforecastskillacrossleadtimes. ACCvalues>0.5upto18months,comparabletotheXROand3D-
Geoformer. By contrast, DESN outperformed all baselines, main-
Results tainingACC>0.5acrossleadtimesupto20months,withparticularly
Toexplorelong-leadENSOpredictability,wedevelopaphysics-guided strongperformanceduringmajorElNiñoepisodes(Figs.S1andS2).
DESN framework that operates on a compact set of physically inter- BeyondENSO,theDESNalsodeliveredskillfullong-leadpredictions
pretableclimatepredictors.Guidedbytheextendedrechargeoscillator for nine other climate modes, achieving lead times of 9–20 months
(XRO)framework25,themodelincorporatesNiño3.4(seasurfacetem- (Fig. S3). Robustness was further demonstrated through 24-year
peratureanomaliesaveragedover170°–120°W,5°S–5°N)34,warmwater leave-outcross-validationacrossthefull1958–2023window,where
volume(WWV),definedas20°Cisothermdepthanomaliesaveraged DESNretainedACC>0.5atleadtimesof11–18months(Fig.S4).In
over 120°E–80°W, 5°S–5°N12, and a suite of key inter-basin climate additiontoforecastaccuracy,DESNofferssubstantialcomputational
modes(Fig.1a).TheseincludetheNorthandSouthPacificMeridional efficiency. It trains in seconds on a standard CPU, with runtimes
Modes(NPMMandSPMM)18,35,theIndianOceanBasin(IOB)mode36, comparabletotheXROwhileachievingforecastskillthatmatchesor
the Indian Ocean Dipole (IOD) ?, the Southern Indian Ocean Dipole exceeds both dynamical and deep learning approaches (Table 1).
(SIOD)37, as well as Tropical North Atlantic (TNA) variability19, the These results show that DESN generalizes well across decades,
Atlantic Niño (ATL3)38, and the South Atlantic Subtropical Dipole avoidsoverfittingtoparticularENSOregimesorclimatestates,and
(SASD)39,allderivedfromtheORAS5oceanreanalysis(see“Methods” providesacomputationallyefficientpathwayforadvancinglong-lead
andTableS1).Together,thesetenphysicallymotivatedclimateindices, climateprediction.
alongwithsinusoidalseasonalcycles,constitutethemodelinputs.Sto-
chastic variability is incorporated as part of the model’s dynamical Mechanismsshapinglong-termENSOpredictability
evolution and addressed through ensemble forecasting to extract the Here we identify the dynamical mechanisms that sustain ENSO pre-
predictablesignal(Fig.1b).Duringinference,themodelgeneratesrolling dictabilityatextendedleadtimes.Usingasuiteoftargetedandcom-
monthlyforecastsupto21monthsahead,initializedateachcalendar putationallyefficientexperimentsenabledbytheDESNframework,we
monthwith20-memberensembles.Forecastskillisevaluatedusingin- show that long-lead ENSO predictability arises from nonlinear, sea-
sample,out-of-sample,and24-yearblockcross-validationschemes(see sonallymodulatedcross-basininteractionsthatareintegratedintothe
Methods),providingarobustbasisforassessinglong-leadENSOpre- subsurface ocean memory through WWV. By systematically varying
dictabilityanditsdynamicalorigins. predictor sets, input dimensionality, and model structure, and by
npjClimateandAtmosphericScience| ( 2026) 9:92 2

| https://doi.org/10.1038/s41612-026-01360-5 |     |     |     |     |     |     | Article |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | ------- | --- |
a
NPMM
TNA
|     |     |     |     | WWV |     |     | ATL3 |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- |
IOD
ENSO
IOB
SPMM
|     |     | SIOD |     |     |     |     | SASD |     |
| --- | --- | ---- | --- | --- | --- | --- | ---- | --- |
b
|     | Input | Reservoir 1 |     |     | Reservoir 2 |     |     |     |
| --- | ----- | ----------- | --- | --- | ----------- | --- | --- | --- |
More layers
Pred phase
Seasonal cycles
Reservoir
Transition*
Concat
Output
*
Fig.1|RegionsofinterestandarchitectureoftheDESNmodelforENSOpre- emulatestochasticvariability.InternalweightsconsistofinputmatricesWl and
in
diction.aStandarddeviationofseasurfacetemperatureanomalies(SSTA)fromthe reservoirmatricesWl .Reservoiroutputsareconcatenatedandmappedtonext-
res
ORAS5reanalysis(1979–2023).Coloredboxesdenoteregionsusedtocompute monthtargetsviaalinearreadoutmatrixW .Multi-stepforecastsareobtained
out
area-averagedclimateindicesforENSOandassociatedmodes,followingtheXRO recursivelyusingrollinginputs.Whenn =1,themodelreducestoastandardESN.
l
framework25.bSchematicoftheDESNarchitecture.Theinputlayercomprises Thereservoirtransitionfunctionatthebottomillustratesneurondynamicswith
multipleclimateindicesandseasonalcyclesasperiodicbootstrapsequences(PBS ). activationfunctiong(⋅)andleakagerateα.SeeMethodsfordefinitionsand
t
Inputsarepassedthroughahierarchyofn recurrentreservoirs,eachgeneratinga implementationdetails.
l
| neuronstatevectorrl.Additivewhitenoiseξl                 |     |     | isinjectedintoeachreservoirto |     |            |     |     |     |
| -------------------------------------------------------- | --- | --- | ----------------------------- | --- | ---------- | --- | --- | --- |
|                                                          | t   |     | rc                            |     |            |     |     |     |
| directlycomparingDESNwiththephysics-basedXROmodel,wedis- |     |     |                               |     | definedas, |     |     |     |
entangletherespectiverolesofseasonalforcing,cross-basincoupling, 2 3
sinωt
| andnonlinearity. | Together, | these analyses | revealhow | physics-guided |     |     |     |     |
| ---------------- | --------- | -------------- | --------- | -------------- | --- | --- | --- | --- |
|                  |           |                |           |                |     | 6   | 7   |     |
constraintsandnonlinearinteractionpathwaysjointlyshapethelimits 6 cosωt 7
|                                          |     |     |     |     |     | ¼6    | 7 ; | ð1Þ |
| ---------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- |
| andsourcesoflong-termENSOpredictability. |     |     |     |     |     | PBS 4 | 5   |     |
t sin2ωt
| First, | we show that seasonal | cycles | can be interpreted | as periodic |     |     |     |     |
| ------ | --------------------- | ------ | ------------------ | ----------- | --- | --- | --- | --- |
cos2ωt
bootstrapsequences.ENSOexhibitsstrongphaselockingtotheseasonal
| cycle, with | events typically | initiating in | boreal spring | and peaking in |     |     |     |     |
| ----------- | ---------------- | ------------- | ------------- | -------------- | --- | --- | --- | --- |
winter41,42.Torepresentthisperiodicity,weintroducedsinusoidalseasonal whereω¼2π¼πistheangularfrequencyoftheannualcycle(T=12)41,t∈
|                                                         |                       |           |        |                       | [0,L−1],andListhenumberoftimestepsintheclimateindices.PBSterms T 6 |     |     |     |
| ------------------------------------------------------- | --------------------- | --------- | ------ | --------------------- | ------------------------------------------------------------------ | --- | --- | --- |
| functions,                                              | or Periodic Bootstrap | Sequences | (PBS), | as additional inputs, |                                                                    |     |     |     |
| npjClimateandAtmosphericScience|  (         2026) 9:92  |                       |           |        |                       |                                                                    |     |     | 3   |

https://doi.org/10.1038/s41612-026-01360-5 Article
a
b
c d e
Fig.2|ForecastperformanceofESNandDESNmodelsforENSOprediction. Ensemble(NMME).IndividualNMMEmodelforecasts(1981-2021)areshownin
aForecastcorrelationskill(ACC)ofthe3-monthrunningmeanNiño3.4indexasa variouscolors.Forecastperiodsforthe3D-GeoformerandNMMEcorrespondto
functionofleadtime,evaluatedout-of-samplefor2002–2023.ShownareDESN theirrespectivetrainingspans.Target-month-dependentACCofNiño3.4forecasts
(red),ESN(magenta),minimal-ESN(darkblue),XRO(black;trainedon fortheminimal-ESN(c),ESN(d),andDESN(e).Colorsdenotecorrelationskillasa
1958–1999),3D-Geoformer(lightblue),andtheIRIoperationalensembleforecasts- functionoftargetmonth(verticalaxis)andleadtime(horizontalaxis).Blackcon-
ensemblemeansofdynamicalmodels(darkpurple)andstatisticalmodels(dark toursdelineateregionswhereACCexceeds0.5,highlightingcombinationsoftarget
cyan).bSameas(a),butevaluatedin-sampleover1979–2023,includingthe monthsandleadtimeswithhighpredictiveskill.
ensemblemeanofdynamicalmodelsfromtheNorthAmericanMulti-Model
areappliedduringbothtrainingandpredictionphases,significantlyaltering Wefurtherinterpretlong-termENSOpredictabilityasarisingfromthe
thereservoirdynamicsandenhancingpredictionaccuracy(Fig.S5).This nonlinear coupling between WWV and additional inter-basin climate
demonstratesthevalueofembeddingknownseasonalstructuredirectlyinto modes,ratherthanbyanysinglepredictoractinginisolation.Thisinter-
themodeltocaptureperiodicforcingmoreeffectively. pretationissupportedbytargetedexperiments(seeMethods).Toclarifythe
npjClimateandAtmosphericScience| ( 2026) 9:92 4

| https://doi.org/10.1038/s41612-026-01360-5 |     |     |     |     |     | Article |
| ------------------------------------------ | --- | --- | --- | --- | --- | ------- |
Table1|ComparisonoftrainingefficiencyacrossENSOforecastingmodels
| Models       | Trainingdata             | Trainingtime |     | Trainingon |     | Months |
| ------------ | ------------------------ | ------------ | --- | ---------- | --- | ------ |
|              | (size)                   | (algorithms) |     |            |     | lead   |
| 3D-Geoformer | Griddedmonthlyseasurface | ~12h         |     | GPU        |     | 13a    |
18b
| 27  | windstressand3Docean | (Backpropagation) |     | (NvidiaV100) |     |     |
| --- | -------------------- | ----------------- | --- | ------------ | --- | --- |
temperatureanomalies(84GB)
| XRO | monthlyclimateindices | 4s                  |     | CPU              |     | 13a |
| --- | --------------------- | ------------------- | --- | ---------------- | --- | --- |
| 25  | (110kB)               | (Linear-regression) |     | (Inteli9-14900K) |     | 18c |
| ESN | monthlyclimateindices | 26s                 |     | CPU              |     | 13a |
18c
|             | (110kB)               | (Ridge-regression) |     | (Inteli9-14900K) |     |     |
| ----------- | --------------------- | ------------------ | --- | ---------------- | --- | --- |
| DESN        | monthlyclimateindices | 122s               |     | CPU              |     | 16a |
| (thisstudy) | (110kB)               | (Ridge-regression) |     | (Inteli9-14900K) |     | 20c |
EstimatedtrainingtimeandcomputationalrequirementsfortheDESN,ESN,XRO,and3D-Geoformermodels.
aOut-of-sampleevaluation(2002–2023).
bOut-of-sample(1983–2021).
cIn-sample(1983–2023).
physicalbasisoftheseexperiments,werecallthatXROrestrictsnonlinear thatlong-leadENSOpredictabilityisgovernedprimarilybythenonlinear
terms to ENSO-WWV coupling and a conditional IOD quadratic con- couplingbetweenWWVandinter-basinclimatemodes.
tribution. In contrast, DESN does not impose explicit constraints on Our results further demonstrate that nonlinear cross-basin interac-
interaction structure, allowing higher-order and cross-basin nonlinear tionsunderpinlong-leadENSOpredictability.Afirstlineofevidenceforthe
couplingstoemergeimplicitlyfromthedata. importanceofnonlinearityemergesfromthemode-decouplingandmode-
Todiagnosethesourcesandcharacteristictimescalesofpredictability, additionexperiments,asrevealedbythesystematiccomparisonbetween
weconductcontrolledmode-decouplingexperimentsinwhichoneclimate XROandDESN(Figs.S6andS7).Althoughbothmodelsaredrivenbythe
modeisremovedfromthefullpredictorsetwhileallremaininginputsare samesetofclimatemodes,DESNconsistentlyoutperformsXROatlong
heldfixed.Figure3a,bpresentsthedecouplingresultsforXROandDESN,
leadtimes,whiletheirskillsremaincomparableatshortleads.Thisdiver-
respectively.Inbothmodels,removingWWVleadstoapronouncedcol- genceemergesdespiteXROalreadyincorporatingmultipleclimatemodes
lapseoflong-leadforecastskill,confirmingitsindispensableroleinsus- and insteadreflects astructural difference:XRO contains onlya limited
taining extended ENSO predictability. By contrast, removing other numberofprescribednonlinearterms,whereasDESNcanrepresentamuch
individualmodeswhileretainingWWVgenerallyaffectsforecastskillbut broader class of nonlinear interactions through its hierarchical reservoir
does not eliminate long-term predictability. Notably, the skill reduction architecture.Thesuperiorlong-leadperformanceofDESNthereforepro-
associatedwithWWVremovalismorepronouncedinDESNthaninXRO videsindirectbutrobustevidencethathigher-ordernonlinearinteractions
(Fig. S6), suggesting that DESN exploits WWV through higher-order contributesubstantiallytoextendedENSOpredictability.
nonlinearinteractionswithotherclimatemodesthatarenotfullyrepre- Asecond,independentlineofevidenceisprovidedbythepronounced
sentedintheprescribedXROframework. seasonalityofforecastskill.Weperformadditionaluninitializedforecast
Tocomplementthedecouplinganalysis,weperformcontrolledmode- experimentsinwhichtheinitialconditionsofselectedinter-basinmodesare
additionexperimentsthatexplicitlyfocusoninteractionsbetweenthecore replacedbyclimatologywhileretainingalllearnedinteractions(seemeth-
ENSOvariablesandindividualexternalclimatemodes.Wedefineabaseline odsandFig.S8).Theseexperimentsrevealastrongseasonalmodulationof
configurationconsistingofNiño3.4andWWV,whichencapsulatesthe findings25.
|                                                                   |     | inter-basin memory, | consistent | with previous      | Notably, | under       |
| ----------------------------------------------------------------- | --- | ------------------- | ---------- | ------------------ | -------- | ----------- |
| classicalrecharge–dischargeframework.Oneadditionalclimatemodefrom |     | specific            |            |                    |          |             |
|                                                                   |     | combinations        | of initial | and target months, | removing | the initial |
thestandardXROsetisthenaddedatatime,andbothXROandDESNare conditionsofAtlanticorIndianOceanmodescanevenimproveforecast
trained under identical input conditions. Figure 3c, d show the mode- skill,indicatingstate-dependentandnonlinearinteractionsthatareabsent
additionresultsforXROandDESN,demonstratingthatretainingWWV inlinearframeworks.Suchbehaviorfurthersupportstheviewthatlong-lead
aloneisinsufficienttosustainlong-leadpredictabilityandthatextended predictability arises from nonlinear, seasonally modulated cross-basin
forecastskillemergesonlywhenWWViscoupledwithadditionalclimate influencesratherthanfromlinearsuperposition.
modes.Amongthese,theNPMMyieldsconsistentimprovementsinboth Finally, we obtain strong mechanistic support by introducing an
XRO and DESN,highlighting its robust contribution to extended-range augmented Sparse-Nonlinear XRO (SN-XRO) model (see methods).
ENSO predictability. Figure 3e, f examines systems of progressively StartingfromthestandardXROformulation,wesystematicallyintroduce
increasing dimensionality, defined by the number of physically distinct quadraticcouplingtermsrepresentingmultiplicativeinteractionsbetween
climatemodescoupledtothecoreENSOvariables(Niño3.4andWWV). Niño3.4,WWV,andexternalclimatemodes,withseasonallymodulated
Predictiveskillincreasesasadditionalclimateindicesarecombinedwith coefficientsestimatedviasparseregression43,44.Whentrainedandevaluated
WWVandNiño3.4,confirmingthatcross-modeinteractionssubstantially under the same conditions as XRO, SN-XRO behaves similarly to the
enhanceENSOpredictability. standardXROatshortleadtimesbutexhibitsapronouncedslowdownin
Takentogether,theseexperimentsprovidecomplementaryevidence skilldegradationatlongerleads(approximately8-19months).Thisdelayed
foraunifiedmechanism.RetainingENSOtogetherwithexternalclimate
improvementcloselymirrorsthebehaviorobservedinDESN,providing
modes while excluding WWV fails to recover long-lead forecast skill, strongsupportingevidencethatenhancednonlinearcross-basincoupling
indicatingthatWWVservesastheprimarymediatorthroughwhichENSO underliestheextensionoftheforecasthorizon(seeFigs.S9-S10).
integrates cross-basin influences at extended lead times. Conversely, Taken together, these results indicate that WWV influences ENSO
retainingWWValonewithinthemode-additionframeworkisinsufficient predictability through slow, nonlinear, and seasonally modulated cross-
tosustainextendedpredictability.ExtendedskillemergesonlywhenWWV basininteractions.ThisinterpretationisconsistentwithestablishedENSO
theory,inwhichinter-basininfluencesareprimarilymediatedbyatmo-
isnonlinearlycoupledwithadditionalinter-basinclimatemodes,withthe
NPMMexertingaparticularlyrobustinfluence.Theseresultsdemonstrate
sphericteleconnectionsandsubsequentlyintegratedintosubsurfaceocean
| npjClimateandAtmosphericScience|  (         2026) 9:92  |     |     |     |     |     | 5   |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |

https://doi.org/10.1038/s41612-026-01360-5 Article
b
a
c d
e f
Fig.3|RoleofWWVandcross-basinmodesinlong-leadENSOpredictability. dashedcurve),withoneadditionalclimatemodeaddedatatime(coloredcurves).
Mode-decouplingexperimentsforXRO(a)andDESN(b).Eachcurveshowsthe ExtendedforecastskillemergesonlywhenWWViscoupledwithadditionalinter-
anomalycorrelationcoefficient(ACC)ofNiño3.4forecastsasafunctionoflead basinmodes,withtheNorthPacificMeridionalMode(NPMM)yieldingparticu-
timewhenoneclimatemodeisremovedfromthefullpredictorsetwhileall larlyrobustimprovements.ThedashedhorizontallineagainindicatesACC=0.5.
remaininginputsareretained.Theblackcurvedenotesthefullmodel,andcolored Increasinginput-dimensionalityexperimentsforWWV-selectedsub-modelsin
curvescorrespondtodecouplingindividualmodes(WWV,NPMM,SPMM,IOB, XRO(e)andDESN(f).Curvesshowtheaverageforecastskillacrossensemblesof
IOD,SIOD,TNA,ATL3,andSASD).RemovingWWVleadstothestrongest sub-modelswithprogressivelyincreasingnumbersofclimatemodescoupledto
degradationoflong-leadskillinbothmodels,whileremovingothermodesgenerally Niño3.4andWWV(from2to9dimensions).Forecastskillincreaseswith
reducesskillwithouteliminatingextendedpredictability.Thehorizontaldashedline inputdimensionality,indicatingthecumulativecontributionofcross-mode
marksACC=0.5.Mode-additionexperimentsforXRO(c)andDESN(d).Forecast interactions.
skillisshownforabaselineconfigurationconsistingofNiño3.4andWWV(black
memory2,3,45.Theconvergenceofevidencefrommodel-structurecompar- perturbations to the initial conditions. Error growth in DESN saturates
isons,seasonalsensitivityexperiments,andexplicitnonlinearaugmentation moreslowlythaninESNorXRO(Fig.4a),confirmingenhanceddynamical
stronglysupportsthecentralroleofnonlinearityinsustaininglong-lead stability. The practical predictabilitylimit was defined as the time when
ENSOpredictability. absoluteerrorreached95%ofitssaturationvalue,estimatedconsistently
acrossmodelsasthemeanabsoluteerrorbetweenmonths60–100ofthe
IntrinsiclimitsofENSOpredictability XROtrajectory.Usingthiscriterion,weobtainpredictabilityhorizonsof
Althoughtheirmethodologicalfoundationsdiffer,arangeofcontemporary approximately18monthsforESN,22monthsforXRO,and34monthsfor
ENSOpredictionframeworks-includingthephysics-basedXROconceptual DESN.Sensitivitytestsfurthershowthatreducingtheinitialperturbation
model,the3D-Geoformerdeeplearningsystem,andourESN/DESNfra- magnitude delays the approach to saturation but yields only limited
meworksconvergeonaforecasthorizonofroughly15–20months.Given extensions of the predictability horizon once the initial error norm falls
thatENSO cyclesspan2-7years,this convergence raisesafundamental below0.1(Fig.4b),indicatingthattheestimatedlimitsarerobust(Fig.4b).
question:doestheplateauinforecastskillreflectmodellimitations,ordoesit Theseresultsindicatethatthewidelycited15–20monthceilingisnot
representanintrinsicbarrierimposedbyENSO’snonlineardynamics? theabsolutebarrierbutratherliesneartheintrinsicpredictabilitylimitof
Tofurtherquantifypredictability,weanalyzethesaturationbehavior ~30months.DESN’sabilitytoapproachthisbarrierarisesnotfromsta-
of forecast errors using an absolute-error metric, inspired by the tisticaloverfittingbutfromitsphysicallyinformeddesign,stabilizedinternal
error-growthframeworkunderlyingtheNLLEapproach.Specifically,we dynamics, and improved representation of multiscale evolution. More
trackthetemporalevolutionofabsolutepredictionerrorsfollowingsmall broadly,theanalysisalsopointstoconcretepathwaysforimprovingforecast
npjClimateandAtmosphericScience| ( 2026) 9:92 6

| https://doi.org/10.1038/s41612-026-01360-5 |     |     |     |     |     |     |     | Article |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ------- |
a b
Fig.4|Error-growthanalysisandnonlinearpredictabilitylimitsofENSO whenabsoluteerrorreaches95%oftheXROsaturationvalue.bAbsoluteerror
models.aEvolutionofabsoluteforecasterrorbetweenperturbedandunperturbed growthinDESNforarangeofinitialperturbationmagnitudesδ.Whilesmaller
0
trajectoriesfortheXRO(black),ESN(green),andDESN(red)models,shownas initialperturbationsdelaytheapproachtosaturation,thelong-termsaturationlevel
lnðδÞasafunctionofleadtime.Dashedverticallinesmarkthepredictabilitylimits andtheinferredpredictabilitylimitremainlargelyunchangedonceδ <0.1.
| i   |     |     |     |     |     |     |     | 0   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
forXRO(22months),ESN(18months),andDESN(34months),definedasthetime
Methods
| skill: reducing | errors in initial | conditions, incorporating |     | more precise |     |     |     |     |
| --------------- | ----------------- | ------------------------- | --- | ------------ | --- | --- | --- | --- |
multivariate couplings across ocean-atmosphere modes, and embedding Dataandpreprocessing
physicallyconsistentconstraintstobettercapturedelayedorcross-basin We use sea surface temperature (SST) and thermocline depth data
feedbacks. fromthe ORAS5 oceanreanalysisdataset(1958–2023)toconstruct
climateindicesrelevanttoENSOdynamics.Specifically,wederiveten
Discussion monthly indices from predefined regions (including the Niño 3.4
regionandtheequatorialPacific20°Cisothermdepth),coveringnine
Ourresultsdemonstratethataphysics-guidedDESNcanextendskillful
ENSO-relatedclimatemodes.Thesemodeswerepreviouslyidentified
| ENSO forecasts | well beyond | one year, maintaining | ACC | above 0.5 for |     |     |     |     |
| -------------- | ----------- | --------------------- | --- | ------------- | --- | --- | --- | --- |
16–20months.Thisperformancematchesorexceedstheskillofoperational
asphysicallylinkedtoENSOvariabilityandarealsofeaturedinthe
explainableandstate-of-the-artdeeplearningmodels,whilerequiringonly extended recharge oscillator (XRO) model25. The definitions and
secondsoftrainingonastandardCPU.Suchefficiencyenablessystematic regions used to calculate these indices are summarized in Supple-
explorationofmodelconfigurationsandphysicalmechanisms,bridgingthe mentaryTableS1.
gapbetweenpredictiveskillandmechanisticunderstanding. ToisolateENSO-relatedvariability,wefirstremovethemonthlycli-
Acentralcontributionofthisstudyisclarifyingthemechanismsthat matologyoverthe1979-2010referenceperiod.Wethenapplyasecond-
sustainlong-leadpredictability.SubmodelexperimentsconfirmthatWWV
orderpolynomialdetrendingtoeliminatelong-termtrends.Theresulting
isthedominantprecursorofENSO,consistentwiththerecharge-discharge standardizedanomaliesrepresentinterannualvariabilityandareusedas
| paradigm,butalsoshowthatitspredictivepowerisamplifiedbycouplings |               |                 |             | modelinputs. |     |     |     |     |
| ---------------------------------------------------------------- | ------------- | --------------- | ----------- | ------------ | --- | --- | --- | --- |
| with remote                                                      | modes such as | the NPMM. These | delayed and | cross-basin  |     |     |     |     |
feedbacksextendpredictability,andDESN’shierarchicalstructureispar- Deepechostatenetworks
ticularlyeffectiveincapturingsuchmultiscaleinteractions. EchoStateNetworks(ESNs)leveragetherichnessofreservoirdynamicsto
Despite the improvedlong-lead forecast skill achieved here,several replicatethetemporalpatternsexpressedintrainingdata.DeepEchoState
Networks(DESNs),firstproposedbyGallicchioetal.30,31,extendclassical
limitationsshouldbenoted.First,theDESNactsasaphysics-guidedsur-
rogate model and does not explicitly resolve coupled ocean-atmosphere ESNsbystackingmultiplerecurrentreservoirlayers.
dynamics,meaningthatphysicalcausalitycannotbedirectlyinferredfrom Ateachtimestept,theDESNprocessesinformationhierarchically:the
thelearnedrepresentations.Second,modelperformancedependsonthe firstreservoirlayerreceivesexternalclimateinputsX ,whilesubsequent
in,t
quality and consistency of the ORAS5 reanalysis used for training, and layerstakeasinputthestatevectorfromtheprecedinglayer.Letn denote
l
uncertaintiesorbiasesintheunderlyingdatamayinfluenceforecastskill. thenumberofreservoirlayers,andNthenumberofneuronsinlayerl—both
l
Third,whilenonlinearerror-growthanalysisprovidesarobustestimateof are architectural hyperparameters set prior to training (see Table S2 for
specificconfigurations).Wedenotethereservoirstateoflayerlattimet
thepredictabilityhorizon,theinferredsaturationtimerepresentsapractical
2RNl.
| ratherthanastricttheoreticallimit.Finally,althoughcross-basininterac- |     |     |     | asrl |     |     |     |     |
| --------------------------------------------------------------------- | --- | --- | --- | ---- | --- | --- | --- | --- |
t
tionsemergeasakeycontributortoextendedpredictability,theirdetailed Toemulatethestochasticnatureofrealclimatedynamics,weintro-
dynamicalpathwaysremaintobeconfirmedthroughtargetednumerical duceadditivewhitenoiseξl withnoiselevelσ intoeachreservoirlayer.The
|     |     |     |     |     |     | rc  | rc  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
experimentswithfullycoupledclimatemodels. stateupdateruleforlayerlisgivenby:
| From | a broader climate | perspective, our results | highlight | physics- |                 |         |         |           |
| ---- | ----------------- | ------------------------ | --------- | -------- | --------------- | ------- | ------- | --------- |
|      |                   |                          |           |          |                 |         | (cid:2) | (cid:3)   |
|      |                   |                          |           |          | ¼ð1(cid:2)αlÞrl | þαltanh | þWl     | þξl ; ð2Þ |
guidedlearningasasurrogatedynamicalmodel,capableofdeliveringboth rl tþ1 Wl Xl rl
|                |                 |                     |                  |     |     | t   | in t res t | rc  |
| -------------- | --------------- | ------------------- | ---------------- | --- | --- | --- | ---------- | --- |
| forecast skill | and mechanistic | clarity. By showing | how WWV-mediated |     |     |     |            |     |
whereαl∈(0,1]istheleakratecontrollingthetime-delaypropertyof
interactionsandmultiscalefeedbacksshapetheintrinsicpredictabilitylimit
of ENSO, this study advances understanding of ENSO as a nonlinear dynamics in layer l, and tanh is the hyperbolic tangent activation
dynamical system. Looking ahead, integrating improved initialization, function ensuring bounded neuron activations. The input weight
richer multivariate observations, and coupled Earth system models may matrix Wl 2RNl×dimðXl Þ is randomly initialized from a uniform
in t
furtherenhance long-leadforecasts.More broadly,thisframeworkillus- distributionandscaledbytheinputscalingparameterσl tocontrolthe
in
|     |     |     |     | strengthofexternalinputs.TherecurrentweightmatrixWl |     |     |     | 2RNl×Nl |
| --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | ------- |
trateshowmachinelearningcanbeembeddedwithindynamicaltheoryto res
radiusρl
advancepredictionandunderstandingofothercriticalcomponentsofthe is constructed as a sparse random matrix with spectral (the
largestabsoluteeigenvalue)andconnectiondensitydl
| climatesystem. |     |     |     |     |     |     |     | (thefractionof |
| -------------- | --- | --- | --- | --- | --- | --- | --- | -------------- |
rc
| npjClimateandAtmosphericScience|  (         2026) 9:92  |     |     |     |     |     |     |     | 7   |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |

| https://doi.org/10.1038/s41612-026-01360-5 |     |     |     |     |     |     |     |     |     |     |     |     |     |     | Article |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- |
nonzeroelements)tomaintaintheechostateproperty28.Thelayerinput We employ ridge regression to determine W by minimizing the
| isdefinedhierarchicallyas: |     |     |     |     |     |     |     |                                    |     |     |     |     | out |     |     |
| -------------------------- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| Xl                         |     |     |     |     |     |     |     | regularizedleast-squaresobjective: |     |     |     |     |     |     |     |
t
(
|     |     |     |     | X    | ;l¼1; |     |     |     | LðW | Þ¼kY(cid:2)W |     | Rk2 þλkW |     | k2 ; | ð8Þ |
| --- | --- | --- | --- | ---- | ----- | --- | --- | --- | --- | ------------ | --- | -------- | --- | ---- | --- |
|     |     |     | ¼   | in;t |       |     | ð3Þ |     |     | out          | out | 2        | out | 2    |     |
Xl
|     |             |     | t        | rl(cid:2)1; | l>1: |        |           |                |                                                             |                                                   |     |     |     |     |     |
| --- | ----------- | --- | -------- | ----------- | ---- | ------ | --------- | -------------- | ----------------------------------------------------------- | ------------------------------------------------- | --- | --- | --- | --- | --- |
|     |             |     |          | t           |      |        |           | 2RN            | (cid:3)Y×T                                                  |                                                   |     |     |     |     |     |
|     |             |     |          |             |      |        |           | whereY(cid:2)P |                                                             | isthetargetmatrixcontainingobservedclimatestates, |     |     |     |     |     |
|     |             |     |          |             |      |        |           | R2R            | n l Nl ×Tistheconcatenatedreservoirstatematrixacrossalltime |                                                   |     |     |     |     |     |
|     |             |     |          |             |      | first  |           |                | l¼ 1                                                        |                                                   |     |     |     |     |     |
| In  | this study, | the | external | input       | X    | to the | reservoir |                |                                                             |                                                   |     |     |     |     |     |
in,t steps and layers, and λ≥0 is the regularization parameter preventing
layercomprisesacomprehensivesetofclimateindicesandseasonal
overfitting.Thisyieldstheclosed-formsolution:
cycles:
|     | 2     |     |     |                   |     |     | 3   |     |     | ¼YR | >ðRR | >þλIÞ(cid:2)1: |     |     | ð9Þ |
| --- | ----- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | ---- | -------------- | --- | --- | --- |
|     | ENSO  |     |     | ðENSOindexÞ       |     |     |     |     |     | W   |      |                |     |     |     |
|     |       | t   |     |                   |     |     |     |     |     | out |      |                |     |     |     |
|     | 6     |     |     | ðWarmWaterVolumeÞ |     |     | 7   |     |     |     |      |                |     |     |     |
|     | 6 WWV |     |     |                   |     |     | 7   |     |     |     |      |                |     |     |     |
|     | 6     | t   |     |                   |     |     | 7   |     |     |     |      |                |     |     |     |
6 ðNorthPacificMeridionalModeÞ 7 I n s u m m a r y , th e D E S N a r c h i t e ct u r e c om b i n e s t h r ee k e y c o m p o n e n t s :
|     | 6 NPMM |     |     |     |     |     | 7   |     |     |     |     |     |     |     |     |
| --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
6 t 7 (1)hi e ra r c hi ca l r es e rv o ir la y er s t h a t e x t ra c t m ult i s c a le t e m p o ra l fe a tu r e s w i t h
|     | 6 SPMM |     | ðSouthPacificMeridionalModeÞ |     |     |     | 7   |     |     |     |     |     |     |     |     |
| --- | ------ | --- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
6 t 7 architecturedefinedbyhyperparametersðn;N;αl;ρl;dl ;σl ;σ Þdetailed
|     | 6   |     |     |     |     |     | 7   |     |     |     |     | l l |     | r c in r c |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- |
6 TNA ðTropicalNorthAtlanticÞ 7 inTable.S2,(2)intrinsicplasticitythatop tim izesactiv a tion di s tributions
|     | 6   | t   |     |     |     |     | 7   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
¼6 ðAtlanticNienoÞ 7 ; ð4Þ withparameters(η,μ,σ),and(3)alinearreadouttrainedviaridgeregression
| X in;t | 6 ATL3 |     |     |     |     |     | 7   |     |     |     |     |     |     |     |     |
| ------ | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t withregularizationλ.Thisdesignmaintainscomputationalefficiencywhile
|     | 6 SASD |     | ðSouthAtlanticSubtropicalDipoleÞ7 |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------ | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 6      | t   |                                   |     |     |     | 7   |     |     |     |     |     |     |     |     |
6 7 achievingstronggeneralizationforlong-leadclimateprediction.
|     | 6   | IOB |     | ðIndianOceanBasinÞ  |     |     | 7   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 6   | t   |     |                     |     |     | 7   |     |     |     |     |     |     |     |     |
|     | 6   |     |     | ðIndianOceanDipoleÞ |     |     | 7   |     |     |     |     |     |     |     |     |
|     | 6   | IOD |     |                     |     |     | 7   |     |     |     |     |     |     |     |     |
|     | 6   | t   |     |                     |     |     | 7   |     |     |     |     |     |     |     |     |
4 SIOD ðSouthIndianOceanDipoleÞ 5 Trainingandvalidationperiods
t
|     |     |     |                                  |     |     |     |     | We employed  | three | complementary |     | experiments |     | to assess | model |
| --- | --- | --- | -------------------------------- | --- | --- | --- | --- | ------------ | ----- | ------------- | --- | ----------- | --- | --------- | ----- |
|     |     | PBS | ðPeriodicbootstrapsequence:Eq:1Þ |     |     |     |     |              |       |               |     |             |     |           |       |
|     |     | t   |                                  |     |     |     |     | performance, |       |               |     |             |     |           |       |
Following reservoir initialization, we apply the Intrinsic Forout-of-sampleforecasting,themodelwastrainedondatafrom
Plasticity (IP) learning rule46 to each reservoir unit to enhance 1958–1999andvalidatedon2002–2023,ensuringthatthevalidationperiod
networkexpressiveness.IPoptimizestheinternalactivationdynamics is temporally independent of the training period. and thus testing the
ofindividualneuronsbyadaptingunit-specificgain(a)andbias(b) model’sabilitytogeneralizetonewperiods.
parameters,suchthatthedistributionofactivationsapproximatesa Forin-sampleforecast,themodelwasbothtrainedandvalidatedon
target Gaussian distribution with target meanμ and target standard the 1979–2023 dataset. This experiment evaluates the model’s ability to
deviationσ. reproduceknowndata,providinginsightintoitsfittingcapacity.
This optimization is formulated as minimizing the Kullback- For Cross-validation, the full 1958–2023 dataset was divided into
Leibler(KL)divergencebetweentheempiricaloutputdistributionand 11subsetsof6consecutiveyears.Foreachiteration,24consecutiveyears
the Gaussian target. For a reservoir unit receiving net input r net , the were withheldfor validation,with the remainderusedfor training.This
activationiscomputedas: rolling-blockdesignensuresthatallportionsoftherecordcontributetoboth
|     |     |     |            |     |      |     |     | training and               | validation, | yielding | a robust | evaluation |     | less dependent | on  |
| --- | --- | --- | ---------- | --- | ---- | --- | --- | -------------------------- | ----------- | -------- | -------- | ---------- | --- | -------------- | --- |
|     |     |     | er¼tanhðar |     | þbÞ; |     | ð5Þ | specifictemporaldivisions. |             |          |          |            |     |                |     |
net
TheTable2summarizesthesevalidationsettings.
wherethegainacontrolstheslopeoftheactivationfunctionandthebiasb
shifts the operating point. At each training step, these parameters are Predictionskillmetrics
updatedaccordingto: Forecastskillwasevaluatedusingtwostandardmetrics:theACCandthe
(cid:4) (cid:3)(cid:5) rootmeansquareerror(RMSE).TheACC,calculatedasthePearsoncor-
e (cid:2)
Δb¼(cid:2)η (cid:2)μ þ r 2σ2þ1(cid:2)er2þμer ; relationbetweenforecasts(f)andobservations(o),
|     |     |     | σ 2 | σ 2 |     |     | ð6Þ |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
P
|     |     |     | Δ ¼ | η þΔb(cid:3)r | ;   |     |     |     |     |     |             | (cid:2)         |            |     |     |
| --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | ----------- | --------------- | ---------- | --- | --- |
|     |     |     | a   |               |     |     |     |     |     |     | ð f (cid:2) | f Þ ð o (cid:2) | (cid:2)o Þ |     |     |
a net ¼qffi ffiffiffi ffiffiffiffiffi ffi ffiffiffiffi ffiffiffiffi ffiffi ffiffi ffiffi ffiffiffi ffiffiffiffiffi ffiffi ffiffiffiffiffiffiffiffiffiffi;
|     |     |     |     |     |     |     |     |     | ACC |     | P                 | P                       |     |     | ð10Þ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | ----------------------- | --- | --- | ---- |
|     |     |     |     |     |     |     |     |     |     |     | ðf (cid:2)(cid:2) | fÞ2 ðo(cid:2)(cid:2)oÞ2 |     |     |      |
whereηistheIPlearningrate.Thisadaptationprocessensuresthateach
reservoirunitmaintainsahigh-entropy,information-richactivationprofile,
(cid:2) and(cid:2)oarethemeansoftheforecastsandobservations,respectively.
| thereby | improving | the | overall | dynamical | capacity | and | generalization | where f |     |     |     |     |     |     |     |
| ------- | --------- | --- | ------- | --------- | -------- | --- | -------------- | ------- | --- | --- | --- | --- | --- | --- | --- |
performanceoftheDESN. TheRMSEmeasurestheaverageforecasterrormagnitude,
After the IP learning process stabilizes the reservoir dynamics, the qffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi
outputoftheDESNiscomputedbyalinearreadoutlayerthatcombines ð11Þ
|     |     |     |     |     |     |     |     |     |     | RMSE | ¼   | ðf (cid:2)oÞ2; |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | -------------- | --- | --- | --- |
statesfromallreservoirlayers:
|     |     |     |     | 2   | 3   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
r1
|     |     |     |     |     | t   |     |     | wheretheoverbardenotesthemean. |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | 6   | 7   |     |     |                                |     |     |     |     |     |     |     |
|     |     |     |     | 6r2 | 7   |     |     |                                |     |     |     |     |     |     |     |
|     |     |     |     | 6   | t 7 |     |     |                                |     |     |     |     |     |     |     |
by ¼W 6 . 7 ; ð7Þ Quantitativeexperimentsforpredictabilitysource
|     |     |     | t   | out4 | . 5 |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
. Torigorouslyquantifythecontributionofindividualclimatemodesand
rnl their interactions to ENSO predictability, we conducted four com-
P t plementarysetsofsystematicexperiments.Allexperimentsusethesame
n
where W 2RNY× l Nl is the readout weight matrix connecting all training(1958–1999)andvalidation(2002–2024)periods.
|     | out |     | l¼ 1 |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
reservoirlayerstotheoutput.Unlikethereservoirweightswhichremain Inthemode-decouplingexperiments,weconsiderninesupplementary
| fixedafterinitialization,W |     |     |     |     |     |     |     |     |     |     |     |     |     | dynamics25 |     |
| -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- |
out istheonlycomponenttrainedusingsuper- climate indices beyond Niño 3.4 that are relevant to ENSO
visedlearning. (Fig. 1a), forming a ten-variable predictor set in the full model. Mode-
| npjClimateandAtmosphericScience|  (         2026) 9:92  |     |     |     |     |     |     |     |     |     |     |     |     |     |     | 8   |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| https://doi.org/10.1038/s41612-026-01360-5 |     |     |     |     |     |     |     |     |     |     |     |     | Article |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- |
Table2|Validationsettingsformodelperformanceassessment
| Validationmethod      |     |     |     |     | Trainingperiod |     |     |     |     |     | Validationperiod |     |     |
| --------------------- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | ---------------- | --- | --- |
| Out-of-sampleforecast |     |     |     |     | 1958–1999      |     |     |     |     |     | 2002–2023        |     |     |
|                       |     |     |     |     | 1979–2023      |     |     |     |     |     | 1979–2023        |     |     |
In-sampleforecast
Cross-validation Remaining7subsets(e.g.,1958–1975,2000–2023) 4consecutivesubsets(e.g.,1976–1999)
decoupling experiments are performed by removing one supplementary theinitialconditionsofallclimatemodeswithinagivenoceanbasinby
modejatatimefromthefullpredictorset,whilekeepingallremaining climatology.Specifically,U NPMM+SPMM removestheinitial-stateinforma-
inputs,modelarchitecture,andtrainingprocedureunchanged. tionofbothPacificMeridionalModes,U removesallIndian
IOB+IOD+SIOD
Thefullmodel,trainedusingalltenclimateindices(Eq.(4)),servesas Oceanmodes,andU removesallAtlanticmodes.
TNA+ATL3+SASD
the control experiment. For example, DESN-D denotes a DESN Theresultingdifferencesinforecastskillisolatethecontributionof
NPMM
experimentinwhichNPMMisexcludedfrombothtrainingandprediction, initial-conditioninformationassociatedwithindividualmodesorentire
withallotherinputsandmodelconfigurationsidenticaltothefullmodel.
oceanbasinstoENSOpredictability.Unlikemode-decouplingexperi-
AnalogousexperimentsareperformedfortheXROframework,denotedas ments,whichremovepredictorsentirelyandalterthedynamicalsystem
XRO-D (e.g.,XRO-D ,XRO-D ).Thedifferenceinforecastskill learned during training, uninitialized experiments retain all learned
|     | j   | WWV |     | NPMM |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
betweenthefullmodelandD quantifiesthesensitivityofENSOpredict- nonlinear interactions and instead diagnose sensitivity to initial-state
j
| abilitytotheremovalofclimatemodej.ComparingDESN-D |           |           |                |     |             | withXRO-D         | uncertainty. |     |     |     |     |     |     |
| ------------------------------------------------- | --------- | --------- | -------------- | --- | ----------- | ----------------- | ------------ | --- | --- | --- | --- | --- | --- |
|                                                   |           |           |                |     |             | j                 | j            |     |     |     |     |     |     |
| under                                             | identical | predictor | configurations |     | reveals the | role of nonlinear |              |     |     |     |     |     |     |
interactionsincross-basininteractions. Sparse-nonlinearXRO(SN-XRO)model
Inthemode-additionexperiments,we startfromabaselinesystem Forreference,thestatevectorofXROisgivenby
| composed | of  | Niño 3.4 | and WWV, | corresponding |     | to the core ENSO |         |     |     |     |     |     |         |
| -------- | --- | -------- | -------- | ------------- | --- | ---------------- | ------- | --- | --- | --- | --- | --- | ------- |
|          |     |          |          |               |     |                  | (cid:2) |     |     |     |     |     | (cid:3) |
recharge-dischargesubsystem.Mode-additionexperimentsareperformed X¼ ;h ;T ;T ;T ;T ;T ;T ;T ;T T:
|     |     |     |     |     |     |     | T ENSO | WWV | NPMM | SPMM | IOB IOD | SIOD TNA | ATL3 SASD |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | ---- | ---- | ------- | -------- | --------- |
byaddingonesupplementaryclimatemodejtothisbaseline,whilekeeping
ð12Þ
allothermodelsettingsunchanged.
| Forexample,DESN-A |     |     |      | denotesaDESNconfigurationinwhich |     |     |                                                         |     |     |     |     |     |     |
| ----------------- | --- | --- | ---- | -------------------------------- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                   |     |     | NPMM |                                  |     |     | IntheoriginalXROformulation,thesystemevolvesaccordingto |     |     |     |     |     |     |
Niño3.4,WWV,andtheNorthPacificMeridionalMode(NPMM)areused
as inputs. Identical experiments are conducted for the XRO framework, dX
|                |            |             |       |         |       |                      |     |     |     | ¼LXþN |     | ðXÞ; | ð13Þ |
| -------------- | ---------- | ----------- | ----- | ------- | ----- | -------------------- | --- | --- | --- | ----- | --- | ---- | ---- |
| denotedasXRO-A |            | (e.g.,XRO-A |       | ,XRO-A  |       | ).                   |     |     |     |       | XRO |      |      |
|                |            | j           |       | NPMM    | SPMM  |                      |     |     | dt  |       |     |      |      |
| The            | difference | in forecast | skill | between | A and | the baseline config- |     |     |     |       |     |      |      |
j
urationquantifiesthesensitivityofENSOpredictabilitytotheinclusionof
wherethenonlineartermsarerestrictedto
climatemodejbeyondthecorerecharge-dischargedynamics.BecauseXRO
isgovernedprimarilybylineardynamicswithprescribedcouplingstruc- 0 1
|     |     |     |     |     |     |     |     |     |     | b T2   | þb T | h        |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | ---- | -------- | --- |
|     |     |     |     |     |     |     |     |     |     | 1 ENSO | 2    | ENSO WWV |     |
ture, systematic skill differences between DESN-A j and XRO-A j under B C
|     |     |     |     |     |     |     |     |     | B   |     | 0   | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
identicalinputshighlightprocessesthatrequirenonlinearrepresentations B C
|                            |     |     |     |     |     |     |     |     | B   |     |     | C   |     |
| -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beyondlinearsuperposition. |     |     |     |     |     |     |     |     | B   |     | 0   | C   |     |
|                            |     |     |     |     |     |     |     |     | B   |     |     | C   |     |
Intheincreasinginput-dimensionexperiments,thebaselineconfig- B 0 C
|     |     |     |     |     |     |     |     |     | B   |     |     | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
u r a ti o n ( d = 2 ) c o ns i st s o fN iñ o 3 .4 a n d W W V , re p r e se n t i n g t h e m in im a l B C
|     |     |     |     |     |     |     |     |     | B   |     | b T2 | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- |
re c h a r ge - di sc h a rg e s y s te m . H i gh e r- d im e ns ion a l c o n fi g u r a t io n s a re c on - N ðXÞ¼ 3 IOD : ð14Þ
|     |     |     |     |     |     |     |     | XRO | B   |     |     | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
structed by sequentially adding supplementary climate modes to this B 0 C
|     |     |     |     |     |     |     |     |     | B   |     |     | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
baseline,yieldinginputdimensionsd=3,4,…,10.Foreachdimensiond,all B 0 C
|     |     |     |     |     |     |     |     |     | B   |     |     | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
possiblecombinationsofd−2modesareselectedfromtheremainingeight B C
|     |     |     |     |     |     |     |     |     | B   |     | 0   | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     |     |     | B   |     |     | C   |     |
supplementary climate indices, while Niño 3.4 and WWV are always @ A
0
retained.Consequently,thed=10configurationcorrespondstothefully
| coupledmodel. |      |                   |     |            |               |                |                               |                    |     |                  | 0   |                   |           |
| ------------- | ---- | ----------------- | --- | ---------- | ------------- | -------------- | ----------------------------- | ------------------ | --- | ---------------- | --- | ----------------- | --------- |
| For           | each | input dimension,  |     | we compute | the mean      | forecast skill | by                            |                    |     |                  |     |                   |           |
|               |      |                   |     |            |               |                | Thus, nonlinearinteractionsin |                    |     | XROareconfinedto |     | ENSOasymmetry,    |           |
| averaging     | over | all corresponding |     | sub-model  | realizations. | This approach  |                               |                    |     |                  |     |                   |           |
|               |      |                   |     |            |               |                | ENSO-WWV                      | recharge-discharge |     | coupling,        |     | and a conditional | quadratic |
isolatesthesystematiceffectofincreasingcouplingdimensionality,inde-
pendentoftheinfluenceofanysingleclimatemode.Identicalincreasing- contributionassociatedwiththeIndianOceanDipole.
IntheSN-XROmodel,weextendthenonlineartendencybyintro-
| dimension | experiments |     | are conducted |     | for both | DESN and XRO |     |     |     |     |     |     |     |
| --------- | ----------- | --- | ------------- | --- | -------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
ducingadditionalquadraticcouplingtermsthatinvolveENSOandWWV:
frameworks.
To investigate state-dependent predictability and the role of initial- 0 1
|     |     |     |     |     |     |     |     |     |     |     | N   | T T |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
condition memory, we performed a set of uninitialized experiments, 1 ENSO M
|     |     |     |     |     |     |     |     |     |     |     | B   | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
denotedasU .Intheseexperiments,theinitialconditionofclimatemodejis B N h T C
|     | j   |     |     |     |     |     |     |     |     |     | B   | 2 wwv M C |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- |
|     |     |     |     |     |     |     |     |     |     |     | B   | C         |     |
replacedbyitsclimatologicalmean(zeroinstandardizedunits),whileall B 0 C
|                                              |     |     |     |     |     |     |     |     |     |     | B   | C   |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| otherinitialconditionsareretainedasobserved. |     |     |     |     |     |     |     |     |     |     | B   | 0 C |     |
|                                              |     |     |     |     |     |     |     |     |     |     | B   | C   |     |
Importantly,alluninitializedexperimentsareconductedusingthefully B C
|     |     |     |     |     |     |     |     |     |     | ðXÞþB |     | 0 C |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- |
trainedDESNmodel,whichistrainedwiththecompletesetofcoupled N ðXÞ¼N B C ; ð15Þ
|     |     |     |     |     |     |     |     | SN  | XRO |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
climatemodes.Thus,thelearnedinteractionstructureremainsunchanged, B 0 C
|     |     |     |     |     |     |     |     |     |     |     | B   | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andonlytheinitial-stateinformationofmodejisremoved.Theimpactof B 0 C
|     |     |     |     |     |     |     |     |     |     |     | B   | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
theinitialconditionofmodejisquantifiedbythedifferenceinforecastskill B C
|                                 |     |     |     |     |     |     |     |     |     |     | B   | 0 C |     |
| ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|                                 |     |     |     |     |     |     |     |     |     |     | B   | C   |     |
| betweenthecontrolexperimentandU |     |     |     | j . |     |     |     |     |     |     | @   | A   |     |
0
Tofurtherdisentangletheroleofbasin-scalememory,wealsocon-
0
ductedocean-leveluninitializedexperimentsbysimultaneouslyreplacing
| npjClimateandAtmosphericScience|  (         2026) 9:92  |     |     |     |     |     |     |     |     |     |     |     |     | 9   |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| https://doi.org/10.1038/s41612-026-01360-5 |     |     |     |     |     |     |     |     |     |     |     |     |     |     | Article |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- |
X0
where areevolvedindependentlyusingthefullnonlineardynamicsforT=
i
100months.
|     | (cid:2) |     |     |     |     |     |     | (cid:3) |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
¼ ;T ;T ;T ;T ;T ;T ;T T; Wetestarangeofinitialperturbationamplitudesϵ .Foreachampli-
| T   | T   |     |     |     |     |     |     |     |     |     |     |     |     | 0   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
M NPMM SPMM IOB IOD SIOD TNA ATL3 SASD tude, we perform 500 realizations with different random perturbation
ð16Þ
directions,sampledfrom500initialconditionsuniformlydistributedacross
thetestperiod(2002–2023).
denotesthesetofexternalclimatemodes.Nononlinearinteractionterms
Theerroratstepiisdefinedas:
areintroducedamongtheexternalclimatemodesthemselves.
|     | Here, N | and N | are seasonally | modulated |     | coefficient | vectors | con- |     |     |       |            |     |     |      |
| --- | ------- | ----- | -------------- | --------- | --- | ----------- | ------- | ---- | --- | --- | ----- | ---------- | --- | --- | ---- |
|     |         | 1     | 2              |           |     |             |         |      |     |     |       | 0          |     |     |      |
|     |         |       |                |           |     |             |         |      |     |     | δ ¼kX | (cid:2)X k | :   |     | ð19Þ |
trolling the n onlinear coupling between ENSO, WWV, and inter-basin i i i 2
coefficient
| climate | modes. | Each |     | is parameterized |     | as a | truncated | Fourier |          |                                                  |     |     |     |     |     |
| ------- | ------ | ---- | --- | ---------------- | --- | ---- | --------- | ------- | -------- | ------------------------------------------------ | --- | --- | --- | --- | --- |
|         |        |      |     |                  |     |      |           |         | where∥⋅∥ | denotestheEuclideannorm(L2norm).Theabsoluteerror |     |     |     |     |     |
| series, |        |      |     |                  |     |      |           |         |          | 2                                                |     |     |     |     |     |
growthisquantifiedbythelogarithmicerror:
| nðtÞ¼n |     | þn sinðωtÞþn |     | cosðωtÞþn | sinð2ωtÞþn |     | cosð2ωtÞ; |      |     |     |       |     |     |     |      |
| ------ | --- | ------------ | --- | --------- | ---------- | --- | --------- | ---- | --- | --- | ----- | --- | --- | --- | ---- |
|        |     |              |     |           |            |     |           |      |     |     | ¼lnδ: |     |     |     | ð20Þ |
|        | 0   | 1s           |     | 1c        | 2s         |     | 2c        |      |     |     | E     |     |     |     |      |
|        |     |              |     |           |            |     |           | ð17Þ |     |     | i     | i   |     |     |      |
define
|     |     |     |     |     |     |     |     |     | To  | aconsistent | saturation | threshold | across | models | with dif- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | --------- | ------ | ------ | --------- |
whereω=2π/12representstheannualcycle.
|     |     |     |     |     |     |     |     |     | ferent error | growth rates, | we compute | the | mean logarithmic | error | over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------------- | ---------- | --- | ---------------- | ----- | ---- |
TheSparseIdentificationofNonlinearDynamicalsystems(SINDy)
months60–100ofXROevolution:
| method43,44 |     |         | fit |                |     |                |     |           |     |     |     |     |     |     |     |
| ----------- | --- | ------- | --- | -------------- | --- | -------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
|             | is  | applied | to  | the parameters | of  | the prescribed |     | quadratic |     |     |     |     |     |     |     |
–
in te ra c t io n t e r m s u si ng O R A S 5 d a t a o v er t h e t r a in in g p e r i o d (1 9 7 9 2 0 0 1 ) . X100
|     |     |     |     |     |     |     | fi  |     |     |     | 1   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
T h is p r o c ed u r e a u to m at ic al ly e li m i n a te s st a ti st i c a lly i ns i g n i ca n t n o n li n e a r E ¼ E : ð21Þ
|     |     |     |     |     |     |     |     |     |     |     | sat 4 | 0 XRO;i |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------- | --- | --- | --- |
terms,yieldingaparsimoniousrepresentationofnonlinearcouplingcon- i¼60
sistentwiththedata.Theresultingseasonallymodulatednonlinearcoupling isdefinedasthefirsttimewhenthe95
ThepredictabilityhorizonT
| parametersareshowninFig.S9. |                                                        |     |     |     |     |     |     |     |               |            | p       |     |          |     |     |
| --------------------------- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ------------- | ---------- | ------- | --- | -------- | --- | --- |
|                             |                                                        |     |     |     |     |     |     |     | percentileofE | isreached: |         |     |          |     |     |
|                             | ToensureacontrolledcomparisonbetweenXROandSN-XRO,model |     |     |     |     |     |     |     |               | sat        |         |     |          |     |     |
|                             |                                                        |     |     |     |     |     |     |     |               |            | (cid:9) |     | (cid:10) |     |     |
parametersareestimatedintwosuccessivesteps. T ¼min ijhEi≥γE ; ð22Þ
First,thestandardXROmodelisfittedusingtheSINDyframework, p i sat
coefficients
including all linear L and the prescribed nonlinear terms where〈⋅〉denotesaverageoverallrealizations,andγ=0.95isthesaturation
ij
| b T2 | , bT   | h        | , and | b T2 . This | step | yields | a physically | con- |            |     |     |     |     |     |     |
| ---- | ------ | -------- | ----- | ----------- | ---- | ------ | ------------ | ---- | ---------- | --- | --- | --- | --- | --- | --- |
| 1    | ENSO 2 | ENSO WWV |       | 3 IOD       |      |        |              |      | threshold. |     |     |     |     |     |     |
sistentbaselinerepresentationofENSOanditslinearcross-basincouplings.
|                    | IntheSN-XROmodel,alllinearcoefficientsL |     |                                      |     |     | arefixedtothevalues |     |      |                  |     |     |     |     |     |     |
| ------------------ | --------------------------------------- | --- | ------------------------------------ | --- | --- | ------------------- | --- | ---- | ---------------- | --- | --- | --- | --- | --- | --- |
|                    |                                         |     |                                      |     |     | ij                  |     |      | Dataavailability |     |     |     |     |     |     |
| obtainedfromtheXRO |                                         |     | fittingandremainunchanged.Additional |     |     |                     |     | non- |                  |     |     |     |     |     |     |
Datasetsandmodelsusedinthispaperarefreelyavailable.ORAS5:https://
lineartermsarethenestimatedusingSINDy. cds.climate.copernicus.eu/datasets/reanalysis-oras5?tab=overview; XRO
ThishierarchicalfittingstrategyensuresthatdifferencesbetweenXRO
modelENSOforecast:https://github.com/senclimate/XRO;3D-Geoformer
| and | SN-XRO | arise solely | from | enhanced | nonlinear | coupling |     | involving |            |           |                                                |     |     |     |     |
| --- | ------ | ------------ | ---- | -------- | --------- | -------- | --- | --------- | ---------- | --------- | ---------------------------------------------- | --- | --- | --- | --- |
|     |        |              |      |          |           |          |     |           | model ENSO | forecast: | https://msdc.qdio.ac.cn/data/metadata-special- |     |     |     |     |
ENSOandWWV,ratherthanfromchangesinlineardynamics.
detail?id=1602252663859298305&otherId=1602252664035459074;
SN-XROisintegrateddeterministicallyandevaluatedoverthesame
NMME:https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/;IRI
out-of-sampleperiod(2002–2024)asXROandDESN,ensuringthatdif-
|     |     |     |     |     |     |     |     |     | ENSO forecast: | https://iri.columbia.edu/our-expertise/climate/forecasts/ |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --------------------------------------------------------- | --- | --- | --- | --- | --- |
ferencesinforecastperformancearisesolelyfromtheinclusionofphysically
enso/current/;Niño3.4IndexfromtheHadISST1.1:https://psl.noaa.gov/
constrainednonlinearinteractionpathways. data/timeseries/month/DS/Nino34/.
Byconstruction,SN-XROprovidesaminimalnonlinearextensionof
XROthatallowsustodirectlytestwhetherenhancedENSO-WWV-inter- Codeavailability
| basin | coupling | is sufficient | to  | reproduce | the | delayed | error growth | and |     |     |     |     |     |     |     |
| ----- | -------- | ------------- | --- | --------- | --- | ------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
ThePythoncodesusedfortheanalysisareavailableonGitHub(https://
extendedpredictabilityidentifiedinDESN.
github.com/zhangzejing/RC-ENSO).
ComputationofnonlinearlocalLyapunovexponents
Received:8November2025;Accepted:16February2026;
| The                                                            | XRO, ESN, | and | DESN | are all autonomous |     | dynamical |            | systems |            |     |     |     |     |     |     |
| -------------------------------------------------------------- | --------- | --- | ---- | ------------------ | --- | --------- | ---------- | ------- | ---------- | --- | --- | --- | --- | --- | --- |
| whoseforwardevolutioncanbeexpressedasz                         |           |     |      |                    |     | t+1 =F(z  | t ),wherez | t isthe |            |     |     |     |     |     |     |
| fullstatevectorandFistheevolutionoperator.Thestatespacediffers |           |     |      |                    |     |           |            |         | References |     |     |     |     |     |     |
acrossmodels:
|     |     |     |     |     |     |     |     |     | 1. Chen,D.,Zebiak,S.E.,Busalacchi,A.J.&Cane,M.A.Animproved |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
8 procedureforElNiñoforecasting:Implicationsforpredictability.
2R10;
|     | ><  | X   |     |     |     | XRO |     |     | Science269,1699–1702(1995). |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- | --- | --- | --- | --- |
(cid:7) t(cid:8)
¼ r;X 2RNþ10; ESNðN ¼20000Þ 2. McPhaden,M.J.,Zebiak,S.E.&Glantz,M.H.ENSOasanintegrating
| z t | >:(cid:7) | t t(cid:8) |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:2) (cid:3) co n ce p t in e a rt h s c i e n c e . S c ie n c e 3 1 4 , 1 74 0 – 1 7 4 5 ( 2 0 06 ).
|     | r1 ;r2 | ;X 2RN1 | þN2 | þ10; DESN | N   | ¼20000;N |     | ¼12000 |                                                             |                  |                      |                |                |                     |     |
| --- | ------ | ------- | --- | --------- | --- | -------- | --- | ------ | ----------------------------------------------------------- | ---------------- | -------------------- | -------------- | -------------- | ------------------- | --- |
|     | t      | t t     |     |           |     | 1        | 2   |        |                                                             |                  |                      | –              |                |                     |     |
|     |        |         |     |           |     |          |     |        | 3. T im m                                                   | e r m an n , A . | e t a l . E l N iñ o | S o u th e r n | O s c il la ti | o n c o m p lexity. |     |
|     |        |         |     |           |     |          |     | ð18Þ   | Nature559,535–545(2018).                                    |                  |                      |                |                |                     |     |
|     |        |         |     |           |     |          |     |        | 4. Zhang,R.-H.,Rothstein,L.M.&Busalacchi,A.J.Originofupper- |                  |                      |                |                |                     |     |
2R10
where X denotes the 10 climate indices, and r, r1, r2 are the oceanwarmingandElNiñochangeondecadalscalesinthetropical
|                        | t   |     |     |     |     |     | t t | t   |                                       |     |     |     |     |     |     |
| ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- |
| reservoirneuronstates. |     |     |     |     |     |     |     |     | PacificOcean.Nature391,879–883(1998). |     |     |     |     |     |     |
Starting from an initial climate state X on the attractor, we 5. Cai,W.etal.ChangingElNiño–SouthernOscillationinawarming
0
introduceasmallperturbationtothe10climateindices:X0 ¼X þδ , climate.Nat.Rev.EarthEnviron.2,628–644(2021).
|     |     |     |     |     |     |     | 0   | 0   | 0   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
whereδ isarandomunitvectorscaledbyinitialamplitudeϵ .ForESN/ 6. Dijkstra,H.A.NonlinearPhysicalOceanography:ADynamical
|     | 0   |     |     |     |     |     | 0   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
DESN, the corresponding reservoir states are initialized from the SystemsApproachtotheLargeScaleOceanCirculationandElNiño.
unperturbed climate state, ensuring perturbations only affect the cli- AtmosphericandOceanographicSciencesLibrary(Springer
matevariables.BoththereferencetrajectoryX andperturbedtrajectory Netherlands,2005),
i
| npjClimateandAtmosphericScience|  (         2026) 9:92  |     |     |     |     |     |     |     |     |     |     |     |     |     |     | 10  |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

https://doi.org/10.1038/s41612-026-01360-5 Article
7. Suarez,M.J.&Schopf,P.S.AdelayedactionoscillatorforENSO.J. 32. Li,X.,Ding,R.&Li,J.Anewtechniquetoquantifythelocal
Atmos.Sci.45,3283–3287(1988). predictabilityofextremeevents:Thebackwardnonlinearlocal
8. Battisti,D.S.&Hirst,A.C.Interannualvariabilityinatropical Lyapunovexponentmethod.Front.Environ.Sci.10,825233
atmosphere-oceanmodel:Influenceofthebasicstate,ocean (2022).
geometryandnonlinearity.J.Atmos.Sci.46,1687–1712(1989). 33. Hou,Z.etal.AsymmetryofthepredictabilitylimitofthewarmENSO
9. Bjerknes,J.AtmosphericteleconnectionsfromtheequatorialPacific. phase.Geophys.Res.Lett.45,12947–12955(2018).
MonthlyWeatherRev.97,163–172(1969). 34. Trenberth,K.E.ThedefinitionofElNiño.Bull.Am.Meteorol.Soc.78,
10. Cane,M.A.&Zebiak,S.E.AtheoryforElNiñoandtheSouthern 2771–2778(1997).
Oscillation.Science228,1085–1087(1985). 35. Chiang,J.C.&Vimont,D.J.AnalogousPacificandAtlanticmeridional
11. Cane,M.A.,Zebiak,S.E.&Dolan,S.C.ExperimentalforecastsofEl modesoftropicalatmosphere–oceanvariability.J.Clim.17,
Niño.Nature321,827–832(1986). 4143–4158(2004).
12. Jin,F.-F.AnequatorialoceanrechargeparadigmforENSO.PartI: 36. Jin,Y.etal.TheindianoceanweakenstheENSOspringpredictability
conceptualmodel.J.Atmos.Sci.54,811–829(1997). barrier:roleoftheindianoceanbasinanddipolemodes.J.Clim.36,
13. Jin,F.-F.,Lin,L.,Timmermann,A.&Zhao,J.Ensemble-mean 8331–8345(2023).
dynamicsoftheENSOrechargeoscillatorunderstate-dependent 37. Jo,H.-S.etal.Southernindianoceandipoleasatriggerfor
stochasticforcing.Geophys.Res.Lett.34(2007). CentralPacificElNiñosincethe2000s.Nat.Commun.13,6965
14. Barnston,A.G.,Tippett,M.K.,L’Heureux,M.L.,Li,S.&DeWitt,D.G. (2022).
Skillofreal-timeseasonalENSOmodelpredictionsduring2002–11:Is 38. Ham,Y.-G.,Kug,J.-S.&Park,J.-Y.TwodistinctrolesofAtlanticSSTs
ourcapabilityincreasing?Bull.Am.Meteorol.Soc.93,631–651 inensovariability:NorthtropicalAtlanticSSTandAtlanticNiño.
(2012). Geophys.Res.Lett.40,4012–4017(2013).
15. Guilyardi,E.,Capotondi,A.,Lengaigne,M.,Thual,S.&Wittenberg,A. 39. Ham,Y.-G.etal.Inter-basininteractionbetweenvariabilityinthe
T.ENSOmodeling:History,progress,andchallenges.InMcPhaden, SouthAtlanticOceanandtheElNiño/SouthernOscillation.Geophys.
M.J.,Santoso,A.&Cai,W.(eds.)ElNiñoSouthernOscillationina Res.Lett.48,e2021GL093338(2021).
ChangingClimate,199–226(AmericanGeophysicalUnion,2020). 40. McPhaden,M.J.A21stcenturyshiftintherelationshipbetween
16. Izumo,T.etal.InfluenceofthestateoftheIndianOceanDipoleonthe ENSOSSTandwarmwatervolumeanomalies.Geophys.Res.Lett.
followingyear’sElNiño.Nat.Geosci.3,168–172(2010). 39,L09706(2012).
17. Larson,S.M.&Kirtman,B.P.ThePacificmeridionalmodeasan 41. Jin,Y.,Liu,Z.,Lu,Z.&He,C.Seasonalcycleofbackgroundinthe
ENSOprecursorandpredictorintheNorthAmericanMultimodel tropicalPacificasacauseofENSOspringpersistencebarrier.
Ensemble.J.Clim.27,7018–7032(2014). Geophys.Res.Lett.46,13371–13378(2019).
18. Zhang,H.,Clement,A.&DiNezio,P.TheSouthPacificMeridional 42. Stuecker,M.F.,Timmermann,A.,Jin,F.-F.,McGregor,S.&Ren,H.-L.
Mode:AmechanismforENSO-likevariability.J.Clim.27,769–783 AcombinationmodeoftheannualcycleandtheElNiño/Southern
(2014). Oscillation.Nat.Geosci.6,540–544(2013).
19. Ham,Y.-G.,Kug,J.-S.,Park,J.-Y.&Jin,F.-F.Seasurface 43. Brunton,S.L.,Proctor,J.L.&Kutz,J.N.Discoveringgoverning
temperatureinthenorthtropicalAtlanticasatriggerforElNiño/ equationsfromdatabysparseidentificationofnonlineardynamical
SouthernOscillationevents.Nat.Geosci.6,112–116(2013). systems.Proc.Natl.Acad.Sci.113,3932–3937(2016).
20. Ding,H.,Keenlyside,N.S.&Latif,M.ImpactoftheequatorialAtlantic 44. deSilva,B.M.etal.Pysindy:aPythonpackageforthesparse
ontheElNiñoSouthernOscillation.Clim.Dyn.38,1965–1972 identificationofnonlineardynamicalsystemsfromdata.J.Open
(2012). SourceSoftw.5,2104(2020).
21. Rodríguez-Fonseca,B.etal.AreAtlanticNiñosenhancingPacific 45. Dijkstra,H.A.Nonlinearphysicaloceanography:adynamicalsystems
ENSOeventsinrecentdecades?Geophys.Res.Lett.36,L20705 approachtothelargescaleoceancirculationandElNiño,vol.532
(2009). (Springer,2005).
22. Cai,W.etal.Pantropicalclimateinteractions.Science363,eaav4236 46. Jaeger,H.,Lukoševičius,M.,Popovici,D.&Siewert,U.Optimization
(2019). andapplicationsofechostatenetworkswithleaky-integrator
23. Ludescher,J.etal.ImprovedElNiñoforecastingbycooperativity neurons.NeuralNetw.20,335–352(2007).
detection.Proc.Natl.Acad.Sci.USA110,11742–11745(2013).
24. Meng,J.etal.Complexity-basedapproachforElNiñomagnitude Acknowledgements
forecastingbeforethespringpredictabilitybarrier.Proc.Natl.Acad. ThisworkwassupportedbytheNationalNaturalScienceFoundationof
Sci.USA117,177–183(2020). China(GrantNo.42575057,T2525011,42450183,12275020,12135003,
25. Zhao,S.etal.ExplainableElNiñopredictabilityfromclimatemode 12205025,42461144209,62333002),theMinistryofScienceand
interactions.Nature630,891–898(2024). TechnologyofChina(2023YFE0109000),theNationalKeyR\&DProgramof
26. Ham,Y.-G.,Kim,J.-H.&Luo,J.-J.Deeplearningformulti-yearENSO China(2025YFF0517203,2025YFF0517304),andStateKeyLaboratoryof
forecasts.Nature573,568–572(2019). InformationPhotonicsandOpticalCommunications(IPOC2023ZJ02).J.F.
27. Zhou,L.&Zhang,R.-H.Aself-attention–basedneuralnetworkfor issupportedbytheFundamentalResearchFundsfortheCentral
three-dimensionalmultivariatemodelinganditsskillfulENSO Universities.
predictions.Sci.Adv.9,eadf2827(2023).
28. Jaeger,H.&Haas,H.Harnessingnonlinearity:Predictingchaotic Authorcontributions
systemsandsavingenergyinwirelesscommunication.Science304, J.M.,J.X.,andJ.F.designedtheresearch.Z.Z.performedtheanalysis.Z.Z.,
78–80(2004). J.M.,Z.Q,W.D.,J.G.,Z.Y.,J.X.,X.C.,W.C.,J.K.,S.H.,andJ.F.preparedthe
29. Jaeger,H.Echostatenetwork.Scholarpedia2,2330(2007). manuscript,discussedresults,andcontributedtowritingthemanuscript.
30. Gallicchio,C.,Micheli,A.&Pedrelli,L.Deepreservoircomputing:a J.M.andJ.F.ledthewritingofthemanuscript.
criticalexperimentalanalysis.Neurocomputing268,87–99(2017).
31. Gallicchio,C.,Micheli,A.&Pedrelli,L.Designofdeepechostate Competinginterests
networks.NeuralNetw.108,33–47(2018). Theauthorsdeclarenocompetinginterests.
npjClimateandAtmosphericScience| ( 2026) 9:92 11

https://doi.org/10.1038/s41612-026-01360-5 Article
Additionalinformation OpenAccessThisarticleislicensedunderaCreativeCommons
SupplementaryinformationTheonlineversioncontains Attribution4.0InternationalLicense,whichpermitsuse,sharing,
supplementarymaterialavailableat adaptation,distributionandreproductioninanymediumorformat,aslong
https://doi.org/10.1038/s41612-026-01360-5. asyougiveappropriatecredittotheoriginalauthor(s)andthesource,
providealinktotheCreativeCommonslicence,andindicateifchanges
Correspondenceandrequestsformaterialsshouldbeaddressedto weremade.Theimagesorotherthirdpartymaterialinthisarticleare
JunMeng,JinghuaXiaoorJingfangFan. includedinthearticle’sCreativeCommonslicence,unlessindicated
otherwiseinacreditlinetothematerial.Ifmaterialisnotincludedinthe
Reprintsandpermissionsinformationisavailableat article’sCreativeCommonslicenceandyourintendeduseisnotpermitted
http://www.nature.com/reprints bystatutoryregulationorexceedsthepermitteduse,youwillneedto
obtainpermissiondirectlyfromthecopyrightholder.Toviewacopyofthis
Publisher’snoteSpringerNatureremainsneutralwithregardto licence,visithttp://creativecommons.org/licenses/by/4.0/.
jurisdictionalclaimsinpublishedmapsandinstitutionalaffiliations.
©TheAuthor(s)2026
npjClimateandAtmosphericScience| ( 2026) 9:92 12