# On the Structure of IV Estimands

**Isaiah Andrews**, MIT Economics
*April 26, 2017*

**JEL Classification:** C13, C26
**Keywords:** Instrumental variables, misspecification

---

## Abstract

When the overidentifying restrictions of the constant-effect linear instrumental variables model fail, common IV estimators converge to different probability limits. I characterize the estimands of two stage least squares, two step GMM, and limited information maximum likelihood as functions of the single-instrument estimands from the just-identified IV regressions which consider each instrument separately. The limited information maximum likelihood estimand is found to be discontinuous on a set of dimension equal to the number of instruments minus one, and to equal the full parameter space on a set of dimension equal to the number of instruments minus two.

---

## 1. Introduction

A wide variety of estimators have been proposed for the constant-effect linear instrumental variables (IV) model, all of which converge to the true parameter value when the model is correctly specified and an instrument relevance condition holds. At the same time, when the IV model is misspecified common IV estimators typically converge to different probability limits.

The goal of this paper is to characterize the behavior of commonly-used estimators under model misspecification in linear IV models with a single endogenous regressor. In particular, the paper considers two-stage least squares (TSLS), two-step generalized method of moments (TSGMM), limited information maximum likelihood (LIML), and continuous updating generalized method of moments (CUGMM). The probability limits (estimands) of TSLS, TSGMM, and LIML are characterized as functions of the estimands in the just-identified models that use one instrument at a time, holding other features of the data generating process fixed. More limited results are derived for the CUGMM estimand.

As is well understood, the TSLS estimand is linear in the single-instrument estimands with linear combination weights summing to one. By contrast, the TSGMM estimand is generally nonlinear, though continuous, in the single-instrument estimands. More surprisingly the LIML estimand is highly nonlinear in the single-instrument estimands and is discontinuous on a set of dimension equal to the number of instruments minus one. If the controls include a constant, I show that the LIML estimand is discontinuous if and only if the vector of single-instrument estimands is such that (a) the TSLS estimand coincides with the ordinary least squares (OLS) estimand and (b) the R² from the reduced-form regression of the outcome on the instruments is greater than the R² from the first-stage regression of the endogenous regressor on the instruments. As the TSLS estimand approaches the OLS estimand from above the LIML estimand diverges to positive infinity, while as the TSLS estimand approaches the OLS estimand from below the LIML estimand diverges to negative infinity. Moreover, when the TSLS and OLS estimands coincide and the reduced-form R² is equal to the first-stage R², the population LIML objective function does not depend on the structural parameter value considered, so the minimizer is the full parameter space.

Analytical results for the CUGMM estimand are more elusive, but the level sets of this estimand (viewed as a function of the vector of single-instrument estimands) have a structure similar to those of LIML, and I find similar behavior for the LIML and CUGMM estimands in a calibration to data from Yogo (2004).

The approach taken in this paper is distinct from that in the literature on heterogeneous treatment effects. A large literature originating with Imbens & Angrist (1994) characterizes the probability limits of IV estimators as combinations of heterogenous treatment effects under exogeneity and monotonicity assumptions. By contrast my approach, based on single-instrument IV estimands, is agnostic about the source and form of misspecification and so can accommodate heterogeneous treatment effects, invalidity of the instruments, or misspecification of the linear functional form. Further, my results apply directly to IV applications which are difficult to cast into the treatment effects framework, for example Yogo (2004). At the same time, however, my results only relate IV estimands to the single-instrument estimands and other statistical objects, rather than to the causal or structural parameters of interest. Hence, by remaining agnostic about the source of misspecification my approach accommodates models beyond the scope of the heterogeneous treatment effect literature but obtains correspondingly weaker results.

Two papers from the literature on heterogeneous treatment effects of particular relevance to my results are Kolesár (2013) and Mogstad et al. (2016). Kolesár (2013) shows that the LIML estimand can lie outside the convex hull of the individual treatment effects in a heterogeneous treatment effect model. Kolesár's results do not imply the discontinuity of the LIML estimand but do suggest peculiar behavior for this quantity, which my results strongly confirm. Mogstad et al. (2016) derive expressions for a wide variety of estimands in terms of the potential outcomes in the treated and untreated states in a heterogenous treatment effect model with a binary treatment. Their results could be used to link the expressions in the present paper to causal effects in that setting, though further exploration of this possibility is left for future work. Other related work includes Hall & Inoue (2003), who examine the large-sample behavior of GMM estimators under misspecification, and Lee (2016), who proposes an asymptotic variance estimator for TSLS in models with heterogenous treatment effects.

In the next section I formally introduce the IV model and define the IV estimands. Section 3 then presents analytical results on the structure of IV estimands, while Section 4 illustrates these results in a calibration to data from Yogo (2004). All proofs are given in the appendix.

---

## 2. The Linear IV Model and Estimands

Suppose we observe a sample of T observations (Yₜ, Xₜ, Zₜ) drawn from distribution F_T, where Yₜ is an outcome variable, Xₜ is a potentially endogenous regressor, and Zₜ is a k×1 vector of instrumental variables. Let us stack these observations into T×1 vectors Y and X with row t equal to Yₜ and Xₜ respectively, and a T×k matrix Z with row t equal to Zₜ′. Suppose the data obey the linear model

> Y = Xβ + ε,

where β is the scalar parameter of interest. Conventional IV methods impose two further restrictions: the instrument relevance condition E[ZₜXₜ] ≠ 0, and the exclusion restriction E[Zₜεₜ] = 0. The model may accommodate additional exogenous regressors Wₜ as well, but for simplicity I will assume any exogenous variables have already been partialled out.¹

A wide variety of estimation schemes have been proposed for linear IV. To accommodate different estimators in a unified framework while allowing the possibility of efficient estimation in models with heteroskedastic, serially correlated, or clustered data, here I treat linear IV as a special case of GMM. In particular, for gₜ(β) = (Yₜ − Xₜβ)Zₜ note that the usual IV identifying assumptions imply the moment restriction E[gₜ(β)] = E[Zₜεₜ] = 0. For g_T(β) = (1/T)∑ₜ gₜ(β) = (1/T) Z′(Y − Xβ) and Ŵ(β) some (potentially parameter-dependent) symmetric positive-definite k×k weighting matrix, a general class of GMM estimators is defined by

> β̂_W = arg min_β Q̂_W(β) = arg min_β g_T(β)′ Ŵ(β) g_T(β)

if this argmin exists and is unique. This paper focuses on four GMM estimators in particular: TSLS, LIML, TSGMM, and CUGMM.

The TSLS estimator is the simplest, and takes Ŵ(β) = ((1/T) Z′Z)⁻¹. This estimator is asymptotically efficient if the IV model is correctly specified and the errors εₜ are homoskedastic and independent across t, but may otherwise be inefficient. The LIML estimator is likewise efficient under correct specification and homoskedasticity but takes

> Ŵ(β) = σ̂⁻²(β) · ((1/T) Z′Z)⁻¹

for

> σ̂²(β) = (1/T)(Y − Xβ)′ M_Z (Y − Xβ) = σ̂²_Y − 2β σ̂_XY + β² σ̂²_X,

where M_Z = I_T − Z(Z′Z)⁻¹Z′, σ̂²_Y = (1/T)Y′M_ZY, σ̂_XY = (1/T)Y′M_ZX, and σ̂²_X = (1/T)X′M_ZX. For brevity of notation I write β̂_TSLS and Q̂_TSLS(β) for the TSLS estimator and objective, respectively, and do the same for the other estimators considered.

TSGMM attempts to improve efficiency by taking into account the asymptotic variance matrix of the moment vector. In particular, for a given first step estimator β̂₁ of β, which for concreteness I will take to be the TSLS estimator, TSGMM sets Ŵ(β) = Σ̂(β̂₁)⁻¹ for Σ̂(β) an estimator of the asymptotic variance of √T g_T(β). Such variance estimators can typically be decomposed as

> Σ̂(β) = Γ̂₁₁ − β(Γ̂₁₂ + Γ̂₁₂′) + β²Γ̂₂₂

for some k×k matrix-valued estimators Γ̂₁₁, Γ̂₁₂, Γ̂₂₂. I limit attention to variance estimators Σ̂(β) of this form. Finally CUGMM bears the same relationship to TSGMM as TSLS does to LIML and takes Ŵ(β) = Σ̂(β)⁻¹.

### 2.1 Linear IV Estimands

I am interested in the asymptotic behavior of IV estimators, and in particular their estimands. As a starting point I assume that the terms which enter the GMM objective function all tend to well-defined probability limits.

**Assumption 1.** As the sample size T tends to infinity,

> ((1/T)Z′Y, (1/T)Z′X) →ₚ (E[ZₜYₜ], E[ZₜXₜ]).

Moreover,

> (1/T) Z′Z →ₚ E[ZₜZₜ′],  (1)

> [σ̂²_Y, σ̂_XY; σ̂_XY, σ̂²_X] →ₚ [σ²_Y, σ_XY; σ_XY, σ²_X]  (2)

where

> σ²_Y = E[Yₜ²] − E[ZₜYₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜYₜ]
> σ_XY = E[XₜYₜ] − E[ZₜYₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]
> σ²_X = E[Xₜ²] − E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]

and

> [Γ̂₁₁, Γ̂₁₂; Γ̂₁₂′, Γ̂₂₂] →ₚ [Γ₁₁, Γ₁₂; Γ₁₂′, Γ₂₂]  (3)

where the limits in (1)–(3) are all full rank. For convenience I further assume that σ_XY ≠ 0.

#### 2.1.1 Parameter Space

The IV estimands considered in this paper can be expressed as functions of

> ψ = (E[ZₜYₜ], E[ZₜXₜ], E[ZₜZₜ′], σ²_Y, σ_XY, σ²_X, Γ₁₁, Γ₁₂, Γ₂₂).  (4)

It is natural to ask what set of values Ψ for ψ can arise in practice. To explore this question, it is helpful to consider the characterization of the IV model in terms of the reduced-form and first-stage regressions

> Yₜ = Zₜδ + U_{Y,t},  (5)
> Xₜ = Zₜπ + U_{X,t},  (6)

where δ and π are the OLS regression coefficients of Yₜ and Xₜ on Zₜ, respectively. To relate this model to ψ, note that E[ZₜYₜ] = E[ZₜZₜ′]δ and E[ZₜXₜ] = E[ZₜZₜ′]π. Assumption 1 implies that

> [σ²_Y, σ_XY; σ_XY, σ²_X] = E[U²_{Y,t}, U_{Y,t}U_{X,t}; U_{Y,t}U_{X,t}, U²_{X,t}],

so the LIML estimator depends on the second-moment matrix of the residuals from the reduced-form and first-stage regressions. Analogously, depending on the estimator used, (Γ₁₁, Γ₁₂, Γ₂₂) will typically correspond to the asymptotic variance of either (√(1/T) Z′Y, √(1/T) Z′X) or (√(1/T) Z′U_{Y,t}, √(1/T) Z′U_{X,t}).

The constant-effect linear instrumental variables model implies that δ = πβ, but this restriction may fail for a variety of reasons. For example, while we treat Zₜ as an instrument some elements of Zₜ may in fact be exogenous variables which should be included as controls (that is, these elements should be included in Wₜ). Alternatively, it may be that Zₜ is a valid instrument but the true structural relationship is nonlinear, Yₜ = g(Xₜ) + εₜ. In this case

> δ = E[ZₜZₜ′]⁻¹ E[Zₜ g(Xₜ)],

so δ will depend on the true functional form g(Xₜ). Finally, we may have δ ≠ πβ due to heterogenous treatment effects. The exact relationship between the coefficients δ and the underlying heterogeneous effects in such cases is beyond the scope of this paper, but can be derived using the results of Mogstad et al. (2016).

When misspecification is due to misclassification of exogenous variables as instruments we can obtain any value for (δ, π, σ²_Y, σ_XY, σ²_X, E[ZₜZₜ′]) such that E[ZₜZₜ′] and the variance matrix for (U_{Y,t}, U_{X,t}) are positive semi-definite.² Many cases of functional form misspecification and treatment effect heterogeneity, by contrast, will impose additional restrictions on the possible values of ψ. The possible values of (Γ₁₁, Γ₁₂, Γ₂₂) are also restricted, where the exact restrictions depend on the setting and the estimator used. Most of the results of this paper focus on varying E[ZₜYₜ], or equivalently δ, while holding the remaining elements of ψ constant, but in settings that imply additional restrictions one can limit attention to the corresponding parameter space Ψ.

#### 2.1.2 Consistency and Estimands

If we define g(β; ψ) = E[ZₜYₜ] − E[ZₜXₜ]β as the probability limit of g_T(β) and let W(β; ψ) denote the probability limit of Ŵ(β),³ then for the four GMM objective functions discussed above Assumption 1 implies that

> Q̂_W(β) →ₚ Q_W(β; ψ) = g(β; ψ)′ W(β; ψ) g(β; ψ)

for each fixed β. I define the IV estimand β_W(ψ) to be the minimizer of Q_W(·; ψ).

**Definition 1.** Define the IV estimand for weight matrix Ŵ as

> β_W(ψ) = arg min_{β∈ℝ} Q_W(β; ψ).

Note that β_W(ψ) may be set-valued if Q_W(β; ψ) has multiple minimizers. If lim_{β→±∞} Q_W(β; ψ) = inf_β Q_W(β; ψ) then {−∞, +∞} ⊆ β_W(ψ).

To connect β_W(ψ) to the asymptotic behavior of β̂_W, note that if we define

> ψ̂ = ((1/T)Z′Y, (1/T)Z′X, (1/T)Z′Z, σ̂²_Y, σ̂_XY, σ̂²_X, Γ̂₁₁, Γ̂₁₂, Γ̂₂₂)

then for the estimators discussed here β̂_W = β_W(ψ̂). Thus β̂_W →ₚ β_W(ψ) provided β_W(ψ) is a singleton and ψ is a continuity point of β_W(·). Formally:

**Lemma 1.** If β_W(ψ) is a singleton and β_W(·) is continuous on an open neighborhood of ψ, Assumption 1 implies that β̂_W = β_W(ψ̂) →ₚ β_W(ψ).

The focus of this paper is on characterizing the behavior of β_W(ψ) as ψ varies. A well-understood pathology of IV estimators arises when the instrument relevance condition fails and the instruments are orthogonal to the endogenous regressor, E[ZₜXₜ] = 0. For completeness I briefly discuss this case, but will rule it out for the remainder of the paper.

**Irrelevant Instruments.** When E[ZₜXₜ] = 0 the parameter β is not identified even if we assume the exclusion restriction holds. Correspondingly neither Q_TSLS(β) nor Q_TSGMM(β) depends on β,⁴ with the result that β_TSLS(ψ) = β_TSGMM(ψ) = ℝ ∪ {−∞, +∞}. Under the IV exclusion restriction, E[ZₜXₜ] = 0 implies E[ZₜYₜ] = 0 since this is the only way the IV moment condition E[Zₜεₜ] = 0 can hold. In this case we can see that β_LIML(ψ) = β_CUGMM(ψ) = ℝ ∪ {−∞, +∞} as well. If, on the other hand, the IV model is misspecified so E[ZₜXₜ] = 0 but E[ZₜYₜ] ≠ 0, one can show that β_LIML(ψ) = β_CUGMM(ψ) = {−∞, +∞}. To avoid these pathologies, for the remainder of the paper I maintain the instrument relevance assumption. Further, for convenience I assume that each instrument is non-orthogonal to Xₜ.⁵

**Assumption 2.** For each element Z_{i,t} of Zₜ, E[Z_{i,t}Xₜ] ≠ 0.

### 2.2 Single-Instrument IV Estimands

To study the behavior of IV estimands β_W(ψ) under model misspecification it is helpful to have a concise representation for the degree and form of misspecification. One convenient such representation is based on the single-instrument IV estimands.

Note that in a just-identified IV model with k = 1, provided the instrument relevance condition holds, all of the estimands discussed above reduce to β_W(ψ) = E[ZₜYₜ] / E[ZₜXₜ]. Even when k > 1 we can obtain a just-identified system by restricting attention to the ith instrument Z_{i,t}, yielding IV estimand

> bᵢ = E[Z_{i,t}Yₜ] / E[Z_{i,t}Xₜ].

The IV exclusion restriction E[ZₜYₜ] − βE[ZₜXₜ] = 0 holds for some value β if and only if bᵢ = β for all i. Hence, the IV model's overidentifying restrictions hold if and only if bᵢ = bⱼ for all i and j. Denote the set of b = (b₁, ..., b_k)′ such that the IV over-identifying restrictions hold by

> B = {b : bᵢ = bⱼ for all i and j}.  (7)

Note that since b = D(E[ZₜXₜ])⁻¹E[ZₜYₜ] for D(V) which maps the k×1 vector V to a k×k diagonal matrix with the elements of V along the diagonal and zeros elsewhere, we can write

> Q_W(β; ψ) = (b − ιβ)′ Ω(β) (b − ιβ)

for Ω(β) = D(E[ZₜXₜ])′ W(β) D(E[ZₜXₜ]) and ι the k×1 vector of ones. Since correct specification of the IV model restricts only the vector b, to understand the effect of misspecification on IV estimands I will consider behavior as b varies, holding the other elements of ψ (i.e. E[ZₜXₜ], E[ZₜZₜ′], σ²_Y, σ_XY, σ²_X, Γ₁₁, Γ₁₂, Γ₂₂) fixed. See Section 2.1.1 above for further discussion. To emphasize this focus on the behavior of Q_W and β_W in b, I abbreviate Q_W(β; ψ) = Q_W(β; b) and β_W(ψ) = β_W(b) for the remainder of the paper.

---

## 3. The Structure of IV Estimands

As noted in the previous section, all IV estimands coincide in a just-identified model, provided the instrument relevance condition holds. In over-identified models, by contrast, each instrument implies a corresponding IV estimand and the question is how to combine the single-instrument estimands into an overall estimate. The different IV estimators discussed above imply different answers to this question, and the goal of this section is to characterize the behavior of the IV estimands β_W as functions of the single-instrument estimands b.

### 3.1 TSLS

The two stage least squares objective Q_TSLS(β) is quadratic in β, so under Assumption 2 we can solve analytically for

> β_TSLS(b) = [E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜYₜ]] / [E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]] = ∑ᵢ wᵢ bᵢ

where

> wᵢ = [E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ eᵢeᵢ′ E[ZₜXₜ]] / [E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]]

for eᵢ the ith standard basis vector. Note that ∑ᵢ wᵢ = 1 so the TSLS estimand is a linear combination of the single-instrument estimands with weights summing to one. The weights wᵢ are not necessarily positive, however, so we cannot in general interpret β_TSLS(b) as a weighted average of the single-instrument estimands.⁶

### 3.2 TSGMM

The TSGMM objective is again quadratic in β, so we can solve for

> β_TSGMM(b) = ∑ᵢ wᵢ(b) bᵢ

where

> wᵢ(b) = [E[ZₜXₜ]′ Σ⁻¹(β_TSLS(b)) eᵢeᵢ′ E[ZₜXₜ]] / [E[ZₜXₜ]′ Σ⁻¹(β_TSLS(b)) E[ZₜXₜ]].

Thus for a given weight matrix the TSGMM estimand is a linear combination of the single-instrument estimands with weights which sum to one, where the weights themselves now depend on b through the first-stage estimand. The TSGMM estimand will generally be nonlinear in the single-instrument estimands, except in the homoskedastic case where it coincides with TSLS. Assumption 1 implies that Σ(β) is everywhere full rank, and thus that β_TSGMM(b) is a continuous (and, in fact, differentiable) function of b.

### 3.3 LIML

Matters become more interesting when we consider LIML. The well-known characterization of LIML as a k-class estimator (see e.g. Hausman 1983) implies the following.

**Lemma 2.** Let

> Λ(b) = [λ₁(b), λ₂(b); λ₂(b), λ₃] = [b′Ξb, b′Ξι; ι′Ξb, ι′Ξι]

where

> λ₁(b) = E[ZₜYₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜYₜ]
> λ₂(b) = E[ZₜYₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]
> λ₃ = E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]

for Ξ = D(E[ZₜXₜ]) E[ZₜZₜ′]⁻¹ D(E[ZₜXₜ]). For φ(b) the largest root of the quadratic equation

> (λ₁(b) + φσ²_Y)(λ₃ + φσ²_X) − (λ₂(b) + φσ_XY)² = 0,

the LIML estimand β_LIML(b) is given by

> β_LIML(b) = [λ₂(b) + φ(b)σ_XY] / [λ₃ + φ(b)σ²_X]  (8)

whenever the denominator is nonzero.

The only terms in β_LIML(b) which depend on b are λ₂ and φ, both of which change continuously in b. Thus, over most of the parameter space β_LIML will be continuous in b. As might be expected, however, β_LIML behaves strangely when the denominator in (8) crosses zero.

To formally discuss this issue, define the irregular set

> I = {b : λ₃ + φ(b)σ²_X = 0}  (9)

as the set of values b such that the denominator in (8) is zero.

**Proposition 1.** For the irregular set I defined in (9):

1. b ∈ I if and only if λ₁(b)/σ²_Y ≥ λ₂(b)/σ_XY = λ₃/σ²_X.
2. For B the set of single-instrument estimands b such that the IV over-identifying restrictions hold (defined in (7)), B ∩ I = ∅.
3. For b ∈ I,

> β_LIML(b) = {−∞, +∞} if λ₁(b)/σ²_Y > λ₂(b)/σ_XY,
> β_LIML(b) = ℝ ∪ {−∞, +∞} if λ₁(b)/σ²_Y = λ₂(b)/σ_XY = λ₃/σ²_X.

Note that λ₂(b) and λ₁(b) are linear and quadratic in b, respectively. Consequently I consists of the set of points on the (k−1)-dimensional hyperplane {b : λ₂(b) = σ_XY λ₃/σ²_X} lying weakly outside the ellipsoid {b : λ₁(b) = σ²_Y λ₃/σ²_X} centered at the origin. The exclusion of the interior of this ellipsoid ensures that the irregular set does not intersect the parameter space B for the correctly specified IV model, and so reconciles the peculiar behavior of the LIML estimand on I with its well-understood behavior under correct specification. Note that on the set

> S = {b : λ₁(b)/σ²_Y = λ₂(b)/σ_XY = λ₃/σ²_X},

β_LIML(b) is the full extended real line. Hence even though I consider a fixed non-zero value of E[ZₜXₜ], and thus the instrument relevance condition holds, the minimizer of the LIML population objective Q_LIML is not uniquely defined for b ∈ S.

**Interpreting I and S.** To better understand the irregular set I, note that λ₂(b)/σ_XY = λ₃/σ²_X if and only if

> b′Ξι / ι′Ξι = [E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜYₜ]] / [E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]] = σ_XY/σ²_X.  (10)

From the results of Section 3.1 above, however, the left hand side is equal to the TSLS estimand. Using the definitions of σ_XY and σ²_X, however, the OLS estimand from regressing Yₜ on Xₜ can be written as

> β_OLS(b) = E[XₜYₜ] / E[Xₜ²] = (b′Ξι + σ_XY) / (ι′Ξι + σ²_X).

Thus, we see that (10) is equivalent to

> β_TSLS(b) = b′Ξι/ι′Ξι = (b′Ξι + σ_XY)/(ι′Ξι + σ²_X) = β_OLS(b).

Hence, λ₂(b)/σ_XY = λ₃/σ²_X if and only if the TSLS and OLS estimands coincide.

Turning next to the condition that λ₁(b)/σ²_Y ≥ λ₂(b)/σ_XY, note that we can re-write this inequality as

> b′Ξb/σ²_Y ≥ ι′Ξι/σ²_X,

which, using the definitions of σ²_Y and σ²_X, is equivalent to

> [E[ZₜYₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜYₜ]] / E[Yₜ²] ≥ [E[ZₜXₜ]′ E[ZₜZₜ′]⁻¹ E[ZₜXₜ]] / E[Xₜ²].

If the vector of controls Wₜ includes a constant, however, (Xₜ, Yₜ, Zₜ) all have mean zero, and the left hand side is the R² from the reduced-form regression of Yₜ on Zₜ. Likewise, the right hand side is the R² from the first-stage regression of Xₜ on Zₜ. Thus, if the vector of controls includes a constant, λ₁(b)/σ²_Y ≥ λ₂(b)/σ_XY if and only if the R² for the reduced-form exceeds that for the first-stage.

Summing up, if the vector of controls Wₜ includes a constant we see that b ∈ I if and only if (i) β_TSLS(b) = β_OLS(b) and (ii) the reduced-form R² exceeds the first-stage R². Likewise, b ∈ S if and only if β_TSLS(b) = β_OLS(b) and the reduced-form and first-stage have the same R².

**Behavior Outside of I.** It is also interesting to understand the behavior of β_LIML when b ∉ I. To do so, it is helpful to consider the structure of the first-order conditions ∂/∂β Q_LIML(β; b) = 0. We know that if β_LIML(b) = β ∈ ℝ we must have ∂/∂β Q_LIML(β; b) = 0, so if we define

> F_LIML(β) = {b : ∂/∂β Q_LIML(β; b) = 0}

to be the set of values b such that the LIML first order conditions are satisfied for a given β, then the set of b such that β_LIML(b) = β must be a subset of F_LIML(β),

> {b : β_LIML(b) = β} ⊆ F_LIML(β).  (11)

The next lemma characterizes F_LIML(β).

**Proposition 2.** F_LIML(β) is an ellipsoid,

> F_LIML(β) = {b : (b − A(β))′ Ξ (b − A(β)) = C(β)}

where

> A(β) = [(β²σ²_X − σ²_Y) / (−2σ_XY + 2βσ²_X)] ι

and

> C(β) = [(σ²_Y − 2βσ_XY + β²σ²_X) / (−2σ_XY + 2βσ²_X)]² ι′Ξι.

Thus, the set of values b satisfying the LIML first order condition is an ellipsoid with center A(β). Consequently, by (11) the level sets of β_LIML(b) are subsets of ellipsoids. One can confirm that S ⊂ F_LIML(β) for all β, as must be the case given (11) together with the result in part (3) of Proposition 1. Moreover, one can show that any b ∈ I \ S must lie outside of F_LIML(β) in the sense that

> (b − A(β))′ Ξ (b − A(β)) > C(β).

Indeed, for any sequence of single-instrument IV estimands approaching a point in I \ S, the LIML estimand β_LIML(b) diverges.

**Corollary 1.** For any sequence bₙ → b ∈ I \ S such that bₙ ∉ I for all n:

1. limₙ→∞ |β_LIML(bₙ)| → +∞
2. If λ₂(bₙ) > (σ_XY/σ²_X)λ₃ for all n then β_LIML(bₙ) → +∞, while if λ₂(bₙ) < (σ_XY/σ²_X)λ₃ for all n then β_LIML(bₙ) → −∞.

To interpret this result, note that λ₂(b) > (σ_XY/σ²_X)λ₃ implies β_TSLS(b) > β_OLS(b), while the reverse holds for λ₂(b) < (σ_XY/σ²_X)λ₃. Thus, when the TSLS estimand approaches the OLS estimand from above the LIML estimand diverges to +∞, while when the TSLS estimand approaches from below the LIML estimand diverges to −∞.

### 3.4 CUGMM

There is no known closed-form expression for the continuous updating GMM estimator β̂_CUGMM in non-homoskedastic IV models, and the behavior of β_CUGMM(b) is correspondingly harder to characterize. Nonetheless, if one considers the set on which the CUGMM first order conditions are satisfied for a given β

> F_CUGMM(β) = {b : ∂/∂β Q(β; b) = 0}

one can show that these sets are again ellipsoids. Thus, since for β ∈ ℝ,

> {b : β_CUGMM(b) = β} ⊆ F_CUGMM(β),

the level sets of β_CUGMM are again subsets of ellipsoids.

**Proposition 3.** F_CUGMM(β) is an ellipsoid

> F_CUGMM(β) = {b : (b − A*(β))′ B*(β) (b − A*(β)) = C*(β)}

where

> A*(β) = I_k β − Ω⁻¹(β) (−Γ̃₁₂ − Γ̃₁₂′ + 2βΓ̃₂₂)⁻¹ ι
> B*(β) = Ω(β) (−Γ̃₁₂ − Γ̃₁₂′ + 2βΓ̃₂₂) Ω(β)

and

> C*(β) = ι′ (−Γ̃₁₂ − Γ̃₁₂′ + 2βΓ̃₂₂)⁻¹ ι

for Γ̃ᵢⱼ = D(E[ZXₜ])⁻¹ Γᵢⱼ D(E[ZXₜ])⁻¹.

As Proposition 3 highlights, the CUGMM estimand has a structure similar to the LIML estimand, in that the contours of the CUGMM estimand are again subsets of ellipsoids. Unlike in the case of LIML, however, the matrix B*(β) which defines the "shape" of these ellipsoids now depends on β. Moreover, there is not in general a point where all of the sets F_CUGMM(β) intersect, and thus there does not in general exist a value b such that β_CUGMM(b) is equal to the extended real line. Nonetheless, in the next section I find that the CUGMM estimand β_CUGMM(b) exhibits behavior similar to the LIML estimand β_LIML(b) in an example calibrated to data.

---

## 4. IV Estimands in an Example

To illustrate the analytic results above, Figures 1–4 plot the contours of the IV estimands β_W as functions of single-instrument estimands b in a calibration based on Yogo (2004). Yogo studies the effect of weak instruments on estimation of the elasticity of intertemporal substitution using a linear Euler equation model and data from a number of countries. Here I calibrate all elements of ψ other than E[ZₜYₜ] to values estimated from the quarterly US data series used by Yogo, which covers the period from the third quarter of 1947 to the last quarter of 1998. Yogo considers a number of specifications, and here I take the outcome variable Yₜ to be real consumption growth and the endogenous regressor Xₜ to be the real interest rate. Yogo finds that identification-robust Anderson-Rubin confidence sets for the coefficient on the real interest rate are empty in this dataset, suggesting model misspecification.⁷ See Yogo (2004) for further discussion of the data.

Yogo's analysis uses four instruments, namely the two-period lags of consumption growth, the dividend-price ratio, the nominal interest rate, and inflation. In order to plot the contours of β_W(b) here I restrict attention to two instruments, specifically lagged consumption growth and the dividend-price ratio, which I select because they yield easy-to-read plots. Note that while Yogo's analysis is motivated by a concern with weak instruments, here I fix E[ZₜXₜ] at its (non-zero) estimate from the data so weak instruments do not drive the results.

Figures 1–4 bear out the analytical results of the previous section. In particular, we see that the TSLS estimand β_TSLS(b) is linear in b, while the TSGMM estimand is continuous in b but nonlinear. The contours of the LIML estimand are subsets of ellipsoids, and all contours intersect at two points. The CUGMM estimand is in many ways similar to the LIML estimand but its behavior is somewhat more irregular, particularly for small β.

> **Figure 1:** Contours of two-stage least squares estimand β_TSLS(b) as a function of single-instrument estimands b in calibration to US quarterly data from Yogo (2004). *(Linear, evenly-spaced diagonal contour lines.)*

> **Figure 2:** Contours of two-step GMM estimand β_TSGMM(b) as a function of single-instrument estimands b in calibration to US quarterly data from Yogo (2004). *(Continuous but visibly nonlinear/curved contour lines.)*

> **Figure 3:** Contours of limited information maximum likelihood estimand β_LIML(b) as a function of single-instrument estimands b in calibration to US quarterly data from Yogo (2004). *(Contours form ellipsoid-like loops radiating from two intersection points, with values ranging from −30 to 30.)*

> **Figure 4:** Contours of continuously updating GMM estimand β_CUGMM(b) as a function of single-instrument estimands b in calibration to US quarterly data from Yogo (2004). *(Similar to Figure 3, but more irregular.)*

---

## 5. Conclusion

When the over-identifying restrictions of the classical IV model fail, common IV estimators converge to distinct probability limits. Characterizing these limits as a function of the single-instrument IV estimands, I find that the LIML estimand is discontinuous and, further, is sometimes equal to the full parameter space. If the set of controls includes a constant, these issues arise when the OLS and TSLS estimands are equal and the reduced-form R² is weakly larger than the first-stage R². While complete analytical results for CUGMM are more elusive, the contours of the CUGMM estimator resemble those of LIML, and the two estimands display similar behavior in a calibration to data from Yogo (2004).

These results do not necessarily imply that we ought to favor one estimator over another: when single-instrument IV estimands differ, the choice among different IV estimators amounts to a choice of how best to summarize these disparate estimands. Moreover, one might argue that the extreme behavior observed for the LIML estimand is in part an artifact of how we have parameterized the model. If one instead considers the circular parametrization of the IV model as in Chamberlain (2007), for example, then the LIML estimand is continuous on I \ S. Even with this reparameterization, however, the LIML estimand is not uniquely defined on S. Overall, the highly nonlinear and discontinuous behavior of the LIML estimand suggests that caution is warranted when interpreting LIML estimates in misspecified models.

**Acknowledgments.** I am grateful to Jushan Bai, Anna Mikusheva, Whitney Newey, Christoph Rothe, Miikka Rokkanen, Jim Stock, participants in the Fall 2015 Conference in Honor of Jerry Hausman, and two anonymous referees for helpful comments. Support from the Silverman (1978) Family Career Development Chair at MIT and from the National Science Foundation under grant number 1654234 is gratefully acknowledged.

---

## Footnotes

1. Thus, for Ỹ, X̃, W̃, and Z̃ matrices collecting observations of the base variables Ỹₜ, X̃ₜ, W̃ₜ, and Z̃ₜ respectively, I define Y = M_W Ỹ and so on for M_W = I_T − W(W′W)⁻¹W′.
2. In particular, Zₜ is an exogenous control that we have misclassified as an instrument; for β = 0 we can obtain any values of (δ, π, σ²_Y, σ_XY, σ²_X) in (5) and (6).
3. If E[ZₜXₜ] = 0 the weighting matrix Ŵ used by TSGMM typically will not tend to a fixed probability limit, so it will not in general be the case that Q̂_TSGMM(β) converges. Assumption 2 below rules out this case.
4. Here I define β₁(ψ) to be an arbitrary finite singleton so that Q_TSGMM(β) is well-defined.
5. While this is stronger than E[ZₜXₜ] ≠ 0, given an instrument vector Zₜ* such that E[Zₜ*Xₜ] ≠ 0 we can always define a rotation of the instruments Zₜ = OZₜ* such that Assumption 2 holds.
6. In models with heterogenous treatment effects, Angrist & Imbens (1995) and Kolesár (2013) give results which characterize the TSLS estimand as a weighted average of particular causal effects under monotonicity conditions. Their results do not imply, however, that the weights wᵢ above are positive.
7. The Anderson-Rubin confidence set considered by Yogo assumes the data are homoskedastic, and Yogo finds non-empty confidence sets when instead considering the S statistic of Stock & Wright (2000) which relaxes this homoskedasticity assumption. S statistic confidence sets are, however, found to be empty in a linear GMM specification which treats both real interest rates and real equity returns as endogenous regressors.

---

## References

- Angrist, J. & Imbens, G. (1995), 'Two-stage least squares estimation of average causal effects in models with variable treatment intensity', *Journal of the American Statistical Association* 90, 431–442.
- Chamberlain, G. (2007), 'Decision theory applied to an instrumental variable model', *Econometrica* 75, 609–652.
- Hall, A. R. & Inoue, A. (2003), 'The large sample behaviour of the generalized method of moments estimator in misspecified models', *Journal of Econometrics* 114(2), 361–394.
- Hausman, J. (1983), *Handbook of Econometrics*, Vol. 1, North-Holland Publishing Company, chapter Specification and Estimation of Simultaneous Equation Models, pp. 391–448.
- Imbens, G. & Angrist, J. (1994), 'Identification and estimation of local average treatment effects', *Econometrica* 62, 467–475.
- Kolesár, M. (2013), *Estimation in an instrumental variables model with treatment effect heterogeneity*. Unpublished Manuscript.
- Lee, S. (2016), 'A consistent variance estimator for 2SLS when instruments identify different LATEs', *Journal of Business and Economic Statistics* Forthcoming.
- Mogstad, M., Santos, A. & Santos, A. (2016), *Using instrumental variables for inference about policy-relevant treatment effects*. Unpublished Manuscript.
- Stock, J. & Wright, J. (2000), 'GMM with weak identification', *Econometrica* 68, 1055–1096.
- van der Vaart, A. (2000), *Asymptotic Statistics*, Cambridge University Press.
- Yogo, M. (2004), 'Estimating the elasticity of intertemporal substitution when instruments are weak', *Review of Economics and Statistics* 86, 797–810.

---

## Appendix: Proofs

**Proof of Lemma 1.** Follows immediately from the Continuous Mapping Theorem, see e.g. Theorem 2.3 of van der Vaart (2000). ∎

**Proof of Lemma 2.** One may note that the same argument which allows us to derive the expression for β̂_LIML from the objective function Q̂_LIML(·) likewise allows us to derive the expression for β_LIML(ψ) from the objective function Q(·; ψ). For completeness, however, I provide a direct proof.

One can express the LIML estimator as

> β̂_LIML = [X′Y − k̂_LIML X′M_ZY] / [X′X − k̂_LIML X′M_ZX] = [X′P_ZY + (1 − k̂_LIML)X′M_ZY] / [X′P_ZX + (1 − k̂_LIML)X′M_ZX]

for k̂_LIML the smallest root of

> det( [Y′Y, Y′X; Y′X, X′X] − k [Y′M_ZY, Y′M_ZX; Y′M_ZX, X′M_ZX] ) = 0

or, equivalently, φ̂ = (1 − k̂_LIML) the largest root of

> det( [Y′P_ZY, Y′P_ZX; Y′P_ZX, X′P_ZX] + φ[Y′M_ZY, Y′M_ZX; Y′M_ZX, X′M_ZX] ) = (Y′P_ZY + φTσ̂²_Y)(X′P_ZX + φTσ̂²_X) − (Y′P_ZX + φTσ̂_XY)² = 0.

Assumption 1 implies that

> (1/T)[Y′P_ZY, Y′P_ZX; Y′P_ZX, X′P_ZX] →ₚ Λ(b)

for Λ(b) as defined in the text, while

> (1/T)[Y′M_ZY, Y′M_ZX; Y′M_ZX, X′M_ZX] →ₚ [σ²_Y, σ_XY; σ_XY, σ²_X].

Thus, for all φ

> (1/T²)[(Y′P_ZY + φTσ̂²_Y)(X′P_ZX + φTσ̂²_X)] − (1/T²)(Y′P_ZX + φTσ̂_XY)² →ₚ (λ₁ + φσ²_Y)(λ₃ + φσ²_X) − (λ₂ + φσ_XY)².  (12)

Since the largest root of a quadratic equation (when it exists) is a continuous function of the coefficients, and the structure of the problem implies that such a root always exists in the present setting, the Continuous Mapping Theorem implies that φ̂ →ₚ φ for φ the largest root of (12). Thus, again applying the Continuous Mapping Theorem,

> β̂_LIML →ₚ β_LIML(b) = [λ₂(b) + φ(b)σ_XY] / [λ₃ + φ(b)σ²_X]

provided the denominator λ₃ + φ(b)σ²_X is nonzero. ∎

**Proof of Proposition 1.** *Part (1).* Suppose that λ₃ + φ(b)σ²_X = 0. Since φ(b) is the largest root of (12),

> (λ₁(b) + φ(b)σ²_Y)(λ₃ + φ(b)σ²_X) − (λ₂(b) + φ(b)σ_XY)² = 0.

It must therefore be the case that λ₂(b) + φ(b)σ_XY = 0 as well, implying that λ₂(b)/σ_XY = λ₃/σ²_X.

Note, further, that (12) is quadratic in φ and tends to infinity as φ → ±∞ (since positive-definiteness of the limit in (2) implies that σ²_XY < σ²_Xσ²_Y), and thus that a necessary and sufficient condition for a root φ* of (12) to be the largest root (or one of the largest roots if the roots are equal) is that the derivative of (12) at φ* is non-negative. Thus, since the derivative of (12) with respect to φ is

> σ²_Y(λ₃ + φσ²_X) + σ²_X(λ₁(b) + φσ²_Y) − 2σ_XY(λ₂(b) + φσ_XY),

a value φ* with λ₃ + φ*σ²_X = λ₂(b) + φ*σ_XY = 0 is the largest root (or one of the largest roots, if the roots are equal) if and only if

> λ₁(b) + φ*σ²_Y = λ₁(b) − (λ₃/σ²_X)σ²_Y ≥ 0.

Hence the necessary and sufficient condition for λ₃ + φ(b)σ²_X = 0 is that λ₂(b)/σ_XY = λ₃/σ²_X and λ₁(b)/σ²_Y ≥ λ₃/σ²_X.

*Part (2).* For b ∈ B, bᵢ = β̃ for all i, which implies that λ₁(b) = β̃²λ₃ and λ₂(b) = β̃λ₃ for all i, so λ₂(b)/σ_XY = λ₃/σ²_X for b ∈ B if and only if β̃ = σ_XY/σ²_X = ρσ_Y/σ_X (for ρ = σ_XY/(σ_Xσ_Y)). On the other hand λ₁(b)/σ²_Y ≥ λ₂(b)/σ_XY for b ∈ B requires that β̃ ≥ σ_Y/(ρσ_X) if ρ > 0, and that β̃ ≤ σ_Y/(ρσ_X) if ρ < 0. Since I have assumed ρ ≠ 0, these requirements can be satisfied only if ρ = 1 or ρ = −1, which would imply that the right hand side in (2) has reduced rank and so is ruled out by Assumption 1.

*Part (3).* Note that

> Q_LIML(β; b) = [λ₁(b) − 2βλ₂(b) + β²λ₃] / [σ²_Y − 2βσ_XY + β²σ²_X].

Taking the first order condition with respect to β yields that at any local minimum β̃ of Q_LIML, after some algebra,

> σ²_Y(−2λ₂(b) + 2λ₃β̃) − λ₁(b)(−2σ_XY + 2σ²_Xβ̃) + 2β̃²λ₂(b)σ²_X − 2β̃²λ₃σ_XY = 0.  (13)

For b ∈ I, λ₂(b)/σ_XY = λ₃/σ²_X, so (13) becomes

> σ²_Y(λ₃/σ²_X)(−2σ_XY + 2σ²_Xβ̃) − λ₁(b)(−2σ_XY + 2σ²_Xβ̃) = [σ²_Y λ₃/σ²_X − λ₁(b)](−2σ_XY + 2σ²_Xβ̃) = 0.

If σ²_Y λ₃/σ²_X − λ₁(b) = 0 (so λ₁(b)/σ²_Y = λ₃/σ²_X), then this condition holds for all β and Q_LIML(β; b) does not depend on β, implying that β_LIML(b) = ℝ ∪ {−∞, ∞}. If, on the other hand, σ²_Y λ₃/σ²_X − λ₁(b) ≠ 0 then the unique solution to (13) is β̃ = σ_XY/σ²_X. Plugging this back into the LIML objective gives

> Q_LIML(β̃; b) = [λ₁(b) − σ_XY λ₂(b)/σ²_X] / [σ²_Y − σ²_XY/σ²_X] > λ₂(b)/σ_XY = λ₃/σ²_X

where the last inequality follows from the fact that λ₁(b)/σ²_Y > λ₂(b)/σ_XY whenever σ²_Y λ₃/σ²_X − λ₁(b) ≠ 0 and b ∈ I. Since lim_{β→±∞} Q_LIML(β; b) = λ₃/σ²_X, this implies that the unique interior solution to the first order conditions is not a global minimum. Since Q_LIML(β; b) is continuous and everywhere differentiable in β, this is only possible if lim_{β→±∞} Q_LIML(β; b) = inf_β Q_LIML(β; b), from which it follows that β_LIML(b) = {−∞, +∞}. ∎

**Proof of Proposition 2.** This follows immediately from Proposition 3 (Lemma 3 in the original numbering), using the fact that we can recover LIML as a special case of CUGMM by setting

> (Γ₁₁, Γ₁₂, Γ₂₂) = (σ²_Y E[ZₜZₜ′], σ_XY E[ZₜZₜ′], σ²_X E[ZₜZₜ′]).

In this case

> −Γ̃₁₂ − Γ̃₁₂′ + 2βΓ̃₂₂ = (−2σ_XY + 2βσ²_X) Ξ

and

> Ω(β) = Ξ / (σ²_Y − 2βσ_XY + β²σ²_X).

Plugging these expressions into the result of Proposition 3 and dividing through by (−2σ_XY + 2βσ²_X)/(σ²_Y − 2βσ_XY + β²σ²_X) completes the proof. ∎

**Proof of Corollary 1.** Fix a point b ∈ I \ S. Suppose we have a sequence of points bₙ → b, where bₙ ∉ I for all n. I first show that |β_LIML(bₙ)| → ∞.

Define

> R(β; b) = (b − A(β))′ Ξ (b − A(β)) = λ₁(b) − 2a(β)λ₂(b) + a(β)²λ₃  (14)

for a(β) = (β²σ²_X − σ²_Y)/(−2σ_XY + 2βσ²_X), and recall that by Proposition 2, b̃ ∈ F(β) if and only if R(β; b̃) = C(β). Let b* be an arbitrary point in S, where the proof of part (3) of Proposition 1 establishes that b* ∈ F(β) for all β. Note that since b ∈ I \ S,

> R(β; b) − R(β; b*) = λ₁(b) − λ₁(b*) > 0,  (15)

where we have used that b, b* ∈ I implies λ₂(b) = λ₂(b*), and that b* ∈ S, b ∉ S implies λ₁(b) > λ₁(b*).

Thus, the point b lies outside F(β) for all β. Indeed, since (under the norm ‖x‖_Ξ = √(x′Ξx)) F(β) is a circle centered at A(β), the distance from b to F(β) is

> d(b, F(β)) = inf_{b̃∈F(β)} ‖b − b̃‖_Ξ = √R(β;b) − √R(β;b*) = √R(β;b) − √C(β)

where the final equality follows by the definition of F(β).

Note that d(b, F(β)) is a continuous function of β. Thus, for any L > 0

> inf_{β∈[−L,L]} d(b, F(β)) = ε(L) > 0  (16)

since otherwise there must exist some β̃ ∈ [−L, L] with d(b; F(β̃)) = 0, which would contradict (15). Note, further, that ε(L) is decreasing in L by definition. Since bₙ → b, we know that ‖b − bₙ‖_Ξ → 0. Thus, since bₙ ∈ F(β_LIML(bₙ)) by Lemma 2 along with (11), we know that d(b, F(β_LIML(bₙ))) → 0 as n → ∞. Given (16), however, this implies that |β_LIML(bₙ)| → ∞.

Finally, suppose the sequence of points bₙ satisfies λ₂(bₙ) > (σ_XY/σ²_X)λ₃ for all n. I claim that in this case β_LIML(bₙ) → +∞. The proof proceeds by contradiction: suppose β_LIML(bₙ) ↛ +∞. Then by the argument in the previous paragraph there exists a subsequence bₙᵣ such that bₙᵣ → b as r → ∞, λ₂(bₙᵣ) > (σ_XY/σ²_X)λ₃, and β_LIML(bₙᵣ) → −∞. To simplify notation I assume bₙᵣ ≡ bₙ.

Using (14), for b* ∈ S we have that for βₙ = β_LIML(bₙ),

> R(βₙ; bₙ) − R(βₙ, b*) = λ₁(bₙ) − λ₁(b*) − 2a(βₙ)(λ₂(bₙ) − λ₂(b*)).

Since bₙ → b and λ₁(b) > λ₁(b*), there exists an N₁ such that for n > N₁, λ₁(bₙ) > λ₁(b*). Likewise, since a(β) < 0 for β < −σ_Y/σ_X, there exists N₂ such that for n > N₂, a(βₙ) < 0. Thus, since we assumed that λ₂(bₙ) > (σ_XY/σ²_X)λ₃ = λ(b*),

> −2a(βₙ)(λ₂(bₙ) − λ₂(b*)) > 0

which implies that R(βₙ; bₙ) − R(βₙ, b*) > 0 for n ≥ max{N₁, N₂}. This implies that bₙ ∉ F(βₙ), so we have reached a contradiction. Thus, β_LIML(bₙ) → +∞. An argument along the same lines establishes that if λ₂(bₙ) < (σ_XY/σ²_X)λ₃ then β_LIML(bₙ) → −∞, and so completes the proof. ∎

**Proof of Proposition 3.** Note that

> Q_CUGMM(β; b) = (b − ιβ)′ Ω(β) (b − ιβ)

where

> Ω(β) = [Γ̃₁₁ − β(Γ̃₁₂ + Γ̃₁₂′) + β²Γ̃₂₂]⁻¹

and Γ̃ᵢⱼ = D(E[ZXₜ])⁻¹ Γᵢⱼ D(E[ZXₜ])⁻¹. Thus,

> ∂/∂β Q_CUGMM(β; b) = −2ι′Ω(β)(b − ιβ) − (b − ιβ)′ Ω(β)(−Γ̃₁₂ − Γ̃₁₂′ + 2βΓ̃₂₂) Ω(β)(b − ιβ).

Defining

> Ψ(β) = Ω(β)(−Γ̃₁₂ − Γ̃₁₂′ + 2βΓ̃₂₂) Ω(β)

and completing the square in b yields

> ∂/∂β Q_CUGMM(β; b) = −[b − ιβ + Ψ(β)⁻¹Ω(β)ι]′ Ψ(β) [b − ιβ + Ψ(β)⁻¹Ω(β)ι] + ι′Ω(β)Ψ(β)⁻¹Ω(β)ι
> = −[b + (I_kβ − Ψ(β)⁻¹Ω(β))ι]′ Ψ(β) [b + (I_kβ − Ψ(β)⁻¹Ω(β))ι] + ι′(−Γ̃₁₂ − Γ̃₂₁ + 2βΓ̃₂₂)⁻¹ι

which immediately implies the result. ∎
