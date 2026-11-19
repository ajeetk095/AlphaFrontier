# Review of your report vs notebook (read before publishing)

The app uses corrected methodology. These are the places where the submitted report and notebook disagree or aren't supported by the code. Worth knowing before a recruiter reads both.

## Substantive
1. **Objective mismatch.** Report (Eq. 9-12, §3.3) says *maximise Sharpe via SLSQP "via cvxpy"*. Notebook Step 1/2 actually *minimise variance subject to return ≥ the average of the ten asset means*. (SLSQP belongs to SciPy, not cvxpy.) The app uses true max-Sharpe.
2. **Out-of-sample table reuses in-sample numbers.** Notebook sets `annualized_return_1 = ret1` and `sharpe_ratio_1 = sharpe1`, the in-sample values, so Table 6's "Annualised return" (31.49%, 34.14%) and "Sharpe" (2.85, 2.77) are identical to the in-sample tables. The claim "out-of-sample Sharpe ratios are still better than in-sample" is therefore unsupported. Only total return, volatility and drawdown come from test data. Portfolio volatility is a 3-month figure (σ·√n) while the S&P 500's is annualised, so they aren't comparable.
3. **Factor regression window.** Report: full in-sample, 62 weekly observations. Notebook: regresses `perf2`, the *test-period* portfolio returns (Jul-Sep 2025, ~63 days), on *daily* factors, then splits 70/30. Two years of weekly data would be ~104 observations, not 62, and French factors lag so few test-period days may exist. No Huber regression appears in the notebook.
4. **Impossible F-statistic.** With R² = 0.496, k = 5, n = 62: F = (0.496/5)/(0.504/56) ≈ 11.0, not 45.11 (and p ≈ 1e-18 is far too small).
5. **Part 3 uses different inputs than the notebook.** Report Tables 11-17 use *simulated annual returns* (WMT −0.25%, XOM 10.5%…) over "500 trading days"; the notebook uses real 2023-25 data. The 5-asset notebook optimiser is again min-variance, while the simulation maximises Sharpe, so they aren't comparing like with like.
6. **Cap validity rate.** Report says ~86% of draws satisfy the 30% cap. For five assets under a uniform Dirichlet the true probability is ≈ 5.5%, so Table 16's valid-draw counts can't come from that sampler.
7. **Missing variable definitions.** `ret1/vol1/sharpe1/ret2/...` are used but defined in no visible cell; re-run top-to-bottom and confirm.

## Smaller
- Table 5 lists unconstrained HHI 0.1802, but Table 3 (and the weights) give 0.2183.
- Significance legend says `**` = p<0.05, yet HML (p = 0.002) is described as significant "at the 1% level".
- Report says the HML-CMA link shows "quality and value may be the same thing"; CMA is *investment*, RMW is quality.
- First-person comments ("Personally, I think…"), a stray "ítem" in §2.1 and "Step 4" used twice.
- Rubric asks for MLA citations; the reference list is APA-style. Technical-report text mentions tool names (jupyter, cvxpy, SLSQP).
- Sortino and Information Ratio are defined but not reported.

## Suggestion
Present the app as the *improved, reproducible implementation*; keep the report as the submitted academic version. If you want the report corrected, re-run with the app's `core/` functions and update Tables 3-6, 9-10, 11-18.
