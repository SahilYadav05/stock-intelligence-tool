# Step 18D — Selective utility and conformal downside research

Step 18C established that finalized Bank Nifty and India VIX context only marginally improved LONG discrimination and did not improve SHORT. The binary probability objective produced almost no signals and failed proper-score confidence gates.

Step 18D changes the modelling question without changing execution assumptions:

- Predict realized LONG and SHORT R-multiples under the existing 1.0 ATR target, 0.75 ATR stop, 60-minute horizon and conservative costs.
- Compare regularized linear, shallow histogram gradient-boosting and regularized extra-trees regressors.
- Use separate chronological folds for model selection, conformal uncertainty fitting, policy selection and historical diagnosis.
- Convert point forecasts to a pessimistic 20th-percentile utility estimate using residuals from a separate chronological fold.
- Allow a trade only when the pessimistic estimate and directional margin pass the locked policy.
- Require both BUY and SELL support, positive lower-bound expectancy, controlled drawdown and improvement over a NIFTY-only ablation.

This is still historical research. Earlier steps already examined the last historical fold, so even a complete historical pass requires new forward observations before release. The command cannot create an official model artifact, show precise live probabilities, issue an official signal or trade automatically.

No news, futures volume/OI, breadth or external macro history is filled with guessed values. Those remain separate canonical data workstreams.
