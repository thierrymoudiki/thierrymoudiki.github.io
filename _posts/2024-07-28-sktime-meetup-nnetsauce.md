---
layout: post
title: "Copulas for uncertainty quantification in forecasting"
description: "Probabilistic Forecasting with nnetsauce (using Density Estimation, Bayesian inference, Conformal prediction and Vine copulas): nnetsauce presentation at sktime meetup (2024-07-26)"
date: 2024-07-28
categories: Python
comments: true
---

On Friday (2024-07-26), I presented `nnetsauce` ("Probabilistic Forecasting with nnetsauce (using Density Estimation, Bayesian inference, Conformal prediction and Vine copulas)") version `0.23.0` at an [sktime](https://github.com/sktime/sktime) (a unified interface for machine learning with time series) meetup. The news for `0.23.0` are: 

- A method `cross_val_score`: **time series cross-validation** for classes `MTS` and `DeepMTS`, with fixed and increasing window
  
- **Copula simulation** for uncertainty quantification in classes `MTS` and `DeepMTS`: 
  
  - `type_pi` based on copulas of in-sample residuals: `vine-tll` (default), `vine-bb1`, `vine-bb6`, `vine-bb7`, `vine-bb8`, `vine-clayton`, `vine-frank`, `vine-gaussian`, `vine-gumbel`, `vine-indep`, `vine-joe`, `vine-student`
  
  - `type_pi` based on sequential split conformal prediction (`scp`) + vine copula based on [calibrated residuals](https://github.com/thierrymoudiki/2024-07-17-scp-block-bootstrap): `scp-vine-tll`, `scp-vine-bb1`, `scp-vine-bb6`, `scp-vine-bb7`, `scp-vine-bb8`, `scp-vine-clayton`, `scp-vine-frank`, `scp-vine-gaussian`, `scp-vine-gumbel`, `scp-vine-indep`, `scp-vine-joe`, `scp-vine-student`
  
  - `type_pi` based on sequential split conformal prediction (`scp2`) + vine copula based on **standardized** calibrated residuals: `scp2-vine-tll`, `scp2-vine-bb1`, `scp2-vine-bb6`, `scp2-vine-bb7`, `scp2-vine-bb8`, `scp2-vine-clayton`, `scp2-vine-frank`, `scp2-vine-gaussian`, `scp2-vine-gumbel`, `scp2-vine-indep`, `scp2-vine-joe`, `scp2-vine-student`

For more details and examples of use, you can read these slides:

[https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas](https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas)

![xxx]({{base}}/images/2024-07-22/2024-07-22-image1.png){:class="img-responsive"}      

