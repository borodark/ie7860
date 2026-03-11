---
title: Evolution of Time Series Analysis
subtitle: From Classical Forecasting to Deep, Hybrid, and Explainable Models
author: Igor Ostaptchenko
date: March 11, 2026
aspectratio: 169
---

# Evolution of Time Series Analysis

## From Classical Forecasting to Deep, Hybrid, and Explainable Models

- Audience: MS-level students
- Duration: ~60 minutes
- Core framing source: Sahu, *The Evolution of Time Series Analysis: Beyond Traditional Forecasting* (2025)
- Supporting sources: Box-Jenkins, FPP3, Ahmed and Atiya, TFT, deep-learning survey, STGCN, AutoML, XAI literature

---

# What Is a Time Series Problem?

- A time series is an ordered sequence where time carries information
- Forecasting is only one task
- Other tasks include:
  - imputation
  - anomaly detection
  - segmentation
  - classification
  - causal and intervention analysis
- Main challenge: dependence structure changes over time

Examples:

- electricity load
- stock prices
- traffic speed
- ICU vital signs
- retail demand

---

# Why Time Series Is Hard

- autocorrelation violates i.i.d. assumptions
- non-stationarity changes model validity over time
- seasonality can be nested: hourly, daily, weekly, yearly
- shocks create regime changes and structural breaks
- multivariate systems have lagged cross-effects
- evaluation is temporally constrained: future data cannot leak into training

Core question:

> What structure is stable enough to learn, and what structure is changing too fast?

---

# A Short Historical Timeline

- 1970s: Box-Jenkins formalizes ARIMA workflow
- 1980s-2000s: exponential smoothing and state-space formulations mature
- 1990s-2010s: neural nets appear, but are inconsistent in practice
- 2010s: larger data and GPUs make deep sequence models viable
- late 2010s: attention and transformer-based forecasting accelerate
- 2020s: hybrid, AutoML, graph, foundation-style, and explainable forecasting systems emerge

Interpretation:

- the field expanded from "fit one series well" to "build adaptive decision systems over many related series"

---

# Classical Foundation: The Box-Jenkins View

Core ideas from Box, Jenkins, Reinsel, and Ljung:

- identify structure using ACF and PACF
- difference to induce stationarity when needed
- estimate parsimonious ARIMA-family models
- check residuals rather than trusting in-sample fit

Canonical workflow:

1. visualize and transform the series
2. identify candidate orders
3. estimate parameters
4. diagnose residuals
5. forecast and update

Why it mattered:

- it turned forecasting into a disciplined statistical procedure

---

# Classical Foundation: Exponential Smoothing and State Space

Hyndman and Athanasopoulos emphasize:

- level, trend, and seasonality can be modeled explicitly
- exponential smoothing is not just heuristic averaging
- ETS models admit a state-space interpretation

Strengths:

- strong for business forecasting
- efficient with small to moderate datasets
- easy to explain to non-specialists
- often competitive on clean series with regular seasonal structure

---

# Why Classical Methods Still Matter

- they work well when signal structure is regular and data are limited
- they are fast to fit and cheap to maintain
- their assumptions are inspectable
- interval construction and diagnostic checking are mature
- they provide strong baselines that deep models often fail to beat on small data

Practical lesson:

- "old" does not mean obsolete
- in forecasting, baseline discipline is a research skill

---

# Limits of Classical Methods

Classical models often struggle when:

- relationships are strongly nonlinear
- many related series must be modeled jointly
- exogenous covariates are numerous and heterogeneous
- multiple seasonalities interact
- missingness, outliers, and regime shifts are common
- long-range dependencies are important

This pressure created the move toward machine learning and deep learning.

---

# Bayesian Time Series: Where It Fits

Bayesian methods did not disappear in the transition to modern forecasting.

They are especially useful when we need:

- explicit uncertainty quantification, not only point forecasts
- prior knowledge about trend, seasonality, sparsity, or structural relations
- latent-state modeling through dynamic linear or structural time-series models
- coherent updating as new observations arrive
- scenario analysis and intervention assessment

Interpretation:

- Bayesian time series is best seen as a parallel evolution in probabilistic modeling, not a side note between classical and deep methods

---

# Bayesian Applicability and Evidence

Evidence from core Bayesian time-series literature supports several strong use cases:

- Dynamic linear models support trend, seasonal, regression, intervention, and multivariate forecasting in a unified state-space framework
- Bayesian structural time-series models support counterfactual and intervention analysis, as in causal impact estimation
- Bayesian VARs remain important in macroeconomic forecasting because shrinkage helps with high-dimensional multivariate systems
- Recent Federal Reserve work shows fully Bayesian multivariate unobserved-components models can improve out-of-sample trend-cycle forecasts by accounting for state and parameter uncertainty

Best-fit application areas:

- macroeconomic nowcasting and scenario forecasting
- business and policy forecasting under uncertainty
- causal impact analysis for interventions
- systems where probability distributions matter as much as mean forecasts

Caution:

- Bayesian methods can be computationally heavier and sensitive to prior/model specification
- structural breaks can still degrade performance over long forecast windows

---

# Early Machine Learning Shift

The ML shift changed the question from:

- "Which parametric stochastic model generated this series?"

to:

- "Which predictive function best maps past context to future values?"

Typical additions:

- lag feature engineering
- tree ensembles
- kernel methods
- multilayer perceptrons
- support for nonlinear response surfaces

Ahmed and Atiya's empirical study helped legitimize this transition by showing that ML methods could be competitive, especially for nonlinear structure.

---

# What Ahmed and Atiya (2010) Showed

Key message:

- no single model dominates all forecasting problems

Important takeaways:

- MLPs and Gaussian process regression were strong among ML approaches
- model performance depended heavily on data characteristics
- nonlinear methods gained when patterns were irregular or complex
- fair comparison required careful preprocessing, horizons, and evaluation design

MS-level implication:

- benchmarking methodology matters as much as model choice

---

# Deep Sequence Models: RNN, LSTM, and GRU

Why deep sequence models mattered:

- they learn temporal representations instead of relying only on manual lag design

RNN:

- conceptually natural for sequences
- weak at long-range dependency learning

LSTM:

- introduces memory cells and gates
- reduces vanishing-gradient problems

GRU:

- simpler gating than LSTM
- often similar accuracy with lower complexity

---

# When LSTM/GRU Help

Useful when:

- dependencies extend beyond a few lags
- temporal patterns are nonlinear
- many correlated inputs are available
- direct multi-step mapping is more useful than recursive one-step forecasting

But:

- training is slower than classical models
- tuning is sensitive
- gains are not guaranteed on small or clean datasets
- interpretability is weaker without additional tooling

---

# Why Attention Changed the Field

Attention first changed sequence modeling broadly by addressing a core bottleneck:

- not every past time step matters equally

Benefits:

- flexible weighting of relevant history
- better handling of long contexts
- more parallel computation than recurrent-only models
- more interpretable importance patterns than opaque hidden states alone

Conceptual shift:

- sequence modeling moved from compressed recurrence to selective retrieval
- this contribution originated in general sequential learning, then was adapted to forecasting

---

# Transformer Thinking for Time Series

General transformer advantages:

- self-attention captures long-range interactions
- encoder-decoder style supports multi-step prediction
- parallel training improves scalability

Challenges in time series:

- continuous values differ from word tokens
- seasonality and calendar structure must be encoded carefully
- long sequences can be computationally expensive
- forecasting quality depends on horizon design, covariates, and normalization choices

Time-series-specific contributions include:

- long-sequence forecasting variants such as Informer and related efficient attention models
- multi-horizon forecasting with covariates, as in TFT
- stronger global modeling across many related series
- flexible fusion of static, known-future, and observed-past inputs

---

# Temporal Fusion Transformer (TFT)

TFT became influential because it combined:

- recurrent local processing
- attention for longer context
- variable selection networks
- static and time-varying covariate handling
- interpretable importance outputs

Why practitioners care:

- multi-horizon forecasting is realistic for operations
- covariates matter in real systems
- interpretability improves trust compared with generic black-box deep nets

Main lesson:

- modern forecasting architectures are often modular hybrids, not pure transformer stacks
- in time series, transformer contributions are strongest when they are adapted to forecasting structure rather than copied directly from NLP

---

# Forecasting-Specific Transformer Lineage

Representative forecasting papers changed different parts of the design space:

- Informer: improved long-sequence efficiency with ProbSparse attention and one-shot decoding
- Autoformer: pushed decomposition inside the architecture and emphasized periodic dependency discovery
- FEDformer: combined decomposition with frequency-domain modeling to better capture global structure
- PatchTST: treated subseries patches as tokens and improved long-lookback modeling
- TFT: focused on multi-horizon forecasting with heterogeneous covariates and interpretability

Interpretation:

- the contribution is not "Transformer wins"
- the contribution is "forecasting models became architecturally specialized"

---

# Trend and Seasonality Estimation: A Major Design Trend

One of the clearest trends after Informer is explicit structural modeling.

Common pattern:

- separate smoother global movement from faster local fluctuations
- model trend and seasonal or residual components differently
- let the architecture mix statistical decomposition with learned representation

Examples:

- Autoformer uses progressive decomposition inside the network
- FEDformer explicitly argues standard transformers miss the global view, including overall trend
- DLinear, despite being simple, remained competitive by decomposing trend and remainder directly

Research implication:

- time-series progress often comes from injecting temporal structure, not only scaling attention

---

# Current Direction of Travel

Recent model trends suggest four active directions:

- efficient long-horizon architectures for large lookback windows
- decomposition-aware models for trend and seasonality estimation
- patching and representation learning for stronger reusable temporal features
- skepticism toward unnecessary complexity, with simple baselines still forcing recalibration

Inference from the literature:

- the field is converging toward specialized forecasting architectures rather than generic sequence models
- future work will likely mix decomposition, covariate structure, pretraining, and stronger robustness under drift

---

# Deep Learning Did Not "Replace" Statistics

The literature supports a more careful statement:

- deep learning expands the feasible modeling space
- it does not remove the need for baselines, diagnostics, or domain knowledge

Repeated empirical pattern:

- classical methods can remain hard to beat on small, low-noise, strongly seasonal series
- deep models gain more often in large-scale, multivariate, high-complexity settings
- transformer-based methods are important, but their gains are still problem-dependent rather than universal

This is one of the central messages in Lim and Zohren's survey.

---

# Classical vs Deep: A Decision Table

| Dimension | Classical methods | Deep learning methods |
| --- | --- | --- |
| Data requirement | low to moderate | moderate to very high |
| Nonlinearity | limited | strong |
| Interpretability | high | lower, unless designed for XAI |
| Compute cost | low | high |
| Small-data reliability | strong | weaker |
| Multi-series scaling | limited | strong |
| Covariate fusion | modest | strong |
| Deployment simplicity | high | moderate |

Design rule:

- choose the simplest model that matches the real complexity of the system

---

# From Univariate to Multivariate Forecasting

Modern systems rarely involve one isolated series.

Examples:

- traffic sensors influence nearby sensors
- product demand depends on promotions, prices, and holidays
- patient vitals evolve jointly
- energy use depends on weather, occupancy, and calendar effects

New modeling question:

- how should we represent dependencies across variables, nodes, and time scales?

---

# Graph-Based Forecasting

Spatio-temporal graph methods address systems with explicit relational structure.

Example:

- traffic forecasting on a road network

STGCN insight:

- combine graph convolution for spatial dependency
- with temporal convolution for sequence dynamics

Why this matters:

- adjacency is not a nuisance feature
- network topology is part of the forecasting problem itself

---

# Complexity Across Time Scales

Not all signal structure is well described by mean and variance alone.

Multiscale entropy perspective:

- quantify complexity across temporal resolutions
- detect regime changes and shifts in system organization
- useful when the goal is understanding dynamics, not only point prediction

MS-level takeaway:

- forecasting and complexity analysis are related but distinct objectives
- a good analyst knows when prediction accuracy is not the only target

---

# AutoML for Time Series

Why AutoML emerged:

- model search space became too large for manual tuning alone

AutoML can automate:

- preprocessing choices
- lag generation
- model selection
- hyperparameter optimization
- ensembling

Potential benefits:

- faster experimentation
- broader access to forecasting tools
- stronger production baselines

Potential risks:

- hidden leakage
- brittle search objectives
- expensive compute
- false confidence in automated choices

---

# Explainability and XAI

Interpretability is not optional in many domains.

Why:

- forecasts influence inventory, staffing, medicine, finance, and energy operations
- stakeholders need reasons, not only numbers

Common XAI tools in forecasting:

- variable importance
- attention visualization
- SHAP-style local explanation
- counterfactual reasoning
- sensitivity analysis

Caution:

- explanation tools can be persuasive without being faithful

---

# Hybrid Models Are the Real Frontier

Many strong modern systems are hybrids.

Common hybrid patterns:

- decomposition plus neural residual modeling
- statistical baseline plus deep covariate encoder
- local classical model plus global neural model
- expert constraints plus learned representations

Why hybrids work:

- they combine structure, flexibility, and operational robustness

Interpretation:

- the field is evolving toward systems engineering, not model tribalism

---

# A Practical Model Selection Framework

Ask these questions in order:

1. How much data do I have, and at what frequency?
2. Is the series univariate or relational/multivariate?
3. Are covariates available at prediction time?
4. How costly is model failure?
5. How much interpretability is required?
6. How often will the system be retrained?
7. Is the goal point forecast, interval forecast, ranking, or decision support?

If data are scarce:

- start with ETS/ARIMA and strong feature-engineered ML baselines

If scale and complexity are high:

- consider deep, graph, or hybrid models

---

# Mini Case Study: Which Model Would You Choose?

## Case A: Monthly retail demand, 48 observations, clear seasonality

- strong candidate: ETS or seasonal ARIMA

## Case B: 10,000 related SKU series with prices, promotions, and holidays

- strong candidate: global ML/deep model or hybrid system

## Case C: citywide traffic with sensor network topology

- strong candidate: spatio-temporal graph forecasting

## Case D: ICU monitoring with missingness and intervention effects

- strong candidate: multivariate deep/hybrid model with uncertainty-aware evaluation

---

# Evaluation Has Evolved Too

Modern forecasting evaluation should consider:

- rolling-origin validation
- horizon-specific accuracy
- probabilistic metrics, not only point error
- stability under drift
- cost-sensitive decision impact
- latency and retraining cost

Research mistake to avoid:

- choosing a sophisticated model and then evaluating it with an oversimplified protocol

---

# Open Problems

- long-horizon forecasting under distribution shift
- trustworthy uncertainty quantification
- data-efficient deep forecasting
- robust handling of missing and irregular observations
- causal forecasting under interventions
- foundation-style pretraining for temporal data
- fair and faithful explainability
- energy and compute efficiency of large forecasting models

These are active research areas, not solved engineering details.

---

# Key Takeaways

- time series analysis evolved by expanding, not discarding, earlier ideas
- classical methods remain essential because they encode discipline, parsimony, and interpretability
- deep models matter most when complexity, scale, and heterogeneity are real
- transformers and graph models widened the representational toolkit
- AutoML and XAI address usability and trust, but introduce new risks
- the strongest practical systems are increasingly hybrid

---

# Discussion Questions

1. Why do deep models often underperform expectations on small forecasting datasets?
2. Is attention a true explanation or only a useful signal?
3. When should a business prefer a slightly worse but interpretable model?
4. Can AutoML reduce expertise requirements without reducing scientific rigor?
5. What kinds of forecasting problems are fundamentally closer to control than prediction?

---

# Suggested In-Class Exercise

Give each group one forecasting scenario and ask them to justify:

- target variable
- forecast horizon
- evaluation protocol
- baseline model
- advanced model
- explainability requirement
- deployment risk

Deliverable:

- a 3-minute modeling recommendation with explicit tradeoffs

---

# Source Note

This lecture uses Sahu (2025) as a high-level framing paper, but most technical claims should be anchored in stronger primary or widely used sources:

- Box-Jenkins for classical workflow
- Hyndman and Athanasopoulos for forecasting practice
- Ahmed and Atiya for early ML comparison
- Lim and Zohren for deep-learning survey
- Lim et al. for TFT
- Yu et al. for STGCN
- Westergaard et al. for AutoML comparison

Reason:

- survey papers are useful for synthesis, but lecture claims should rest on defensible technical references

---

# References

- Sahu, P. (2025). *The Evolution of Time Series Analysis: Beyond Traditional Forecasting*.
- Ahmed, N. K., and Atiya, A. F. (2010). *An Empirical Comparison of Machine Learning Models for Time Series Forecasting*.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*.
- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*.
- Lim, B., Arik, S. O., Loeff, N., and Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*.
- Lim, B., and Zohren, S. (2021). *Time Series Forecasting With Deep Learning: A Survey*.
- Zhou, H., Zhang, S., Peng, J., et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*.
- Wu, H., Xu, J., Wang, J., and Long, M. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*.
- Zhou, T., Ma, Z., Wen, Q., et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*.
- Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*.
- Zeng, A., Chen, M., Zhang, L., and Xu, Q. (2023). *Are Transformers Effective for Time Series Forecasting?*
- Yu, B., Yin, H., and Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*.
- Costa, M., Goldberger, A. L., and Peng, C.-K. (2005). *Multiscale Entropy Analysis of Biological Signals*.
- Westergaard, G., et al. (2024). *Time Series Forecasting Utilizing Automated Machine Learning (AutoML): A Comparative Analysis Study on Diverse Datasets*.
- Wang, S., Wu, H., Shi, X., et al. (2024). *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*.
- West, M., and Harrison, J. (1997/1999). *Bayesian Forecasting and Dynamic Models*.
- Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., and Scott, S. L. (2015). *Inferring causal impact using Bayesian structural time-series models*.
- Scott, S. L., and Varian, H. R. (2015). *Bayesian Variable Selection for Nowcasting Economic Time Series*.
- Ganics, G., and Odendahl, F. (2021). *Bayesian VAR forecasts, survey information and structural change in the euro area*.
- Jahan-Parvar, M. R., Knipp, C., and Szerszen, P. (2024). *Trend-Cycle Decomposition and Forecasting Using Bayesian Multivariate Unobserved Components*.

---

# Reference Links

- Sahu paper: https://arxiv.org/html/2411.05793v1
- Ahmed and Atiya: https://doi.org/10.1080/07474938.2010.481556
- Box-Jenkins book: https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021
- FPP3 online text: https://otexts.com/fpp3/
- TFT: https://arxiv.org/abs/1912.09363
- Deep-learning survey: https://arxiv.org/abs/2004.13408
- Informer: https://arxiv.org/abs/2012.07436
- Autoformer: https://arxiv.org/abs/2106.13008
- FEDformer: https://arxiv.org/abs/2201.12740
- PatchTST: https://arxiv.org/abs/2211.14730
- DLinear critique: https://arxiv.org/abs/2205.13504
- TimeMixer: https://arxiv.org/abs/2405.14616
- STGCN: https://www.ijcai.org/proceedings/2018/0505.pdf
- Multiscale entropy: https://doi.org/10.1103/PhysRevE.71.021906
- AutoML comparison: https://www.mdpi.com/2078-2489/15/1/39
- Bayesian dynamic models: https://www2.stat.duke.edu/~mw/West%26HarrisonBook/
- BSTS causal impact: https://research.google/pubs/pub41854
- Bayesian nowcasting: https://doi.org/10.3386/w19567
- BVAR and structural change: https://www.bde.es/wbe/en/publicaciones/analisis-economico-investigacion/documentos-trabajo/bayesian-var-forecasts--survey-information-and-structural-change-euro-area.html
- Bayesian trend-cycle decomposition: https://doi.org/10.17016/FEDS.2024.100
