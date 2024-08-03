"""
    BetaNegBinomial(r,α,β)

A *Beta Negative Binomial* is the compound distribution of the [`NegativeBinomial`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.NegativeBinomial) distribution where the probability of success `p` is distributed according to the `Beta`. It has three parameters: `r`, the number of successes number of successes until the experiment is stopped and two shape parameters ``\\alpha``, ``\\beta``
 
```math
P(X = k) = \\frac{B(r + k, \\alpha + \\beta)}{B(r, \\alpha) k!} \\frac{\\Gamma(k + \\beta)}{\\Gamma(\\beta)}
```

```julia
BetaNegBinomial()        # equivalent to BetaNegBinomial(1, 1, 1)
BetaNegBinomial(r)       # equivalent to BetaNegBinomial(r, 1, 1)
BetaNegBinomial(r, α)    # equivalent to BetaNegBinomial(r, α, α)

params(d)        # Get the parameters, i.e. (r , α, β)
succprob(d)    # Get the number of successes, i.e. r
```

External links

* [Beta Negative Binomial distribution on Wikipedia](https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution)
"""
struct BetaNegBinomial{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    r::Int
    α::T
    β::T
    BetaNegBinomial{T}(r, α, β) where {T<:Real} = new{T}(r, α, β)
end

BetaNegBinomial(r::Integer, α::Integer, β::Integer; check_args::Bool=true) = BetaNegBinomial(r, float(α), float(β); check_args=check_args)

function BetaNegBinomial(r::Integer, α::Real, β::Real; check_args::Bool=true)
    @check_args BetaNegBinomial (r, r > zero(r)) (α, zero(α) < α) (β, zero(β) < β)
    return BetaNegBinomial{typeof(α)}(r, α, β)
end

BetaNegBinomial() = BetaNegBinomial{Float64}(1.0, 1.0, 1.0)
BetaNegBinomial(r::Integer) = BetaNegBinomial{Float64}(r, 1.0, 1.0)
BetaNegBinomial(r::Integer, α) = BetaNegBinomial{Float64}(r, α, α)

@distr_support BetaNegBinomial 0 Inf


# parameters
succprob(d::BetaNegBinomial) = d.r

params(d::BetaNegBinomial) = (d.r, d.α, d.β)
partype(::BetaNegBinomial{T}) where {T} = T

#Statistic


Statistics.mean(d::BetaNegBinomial) = d.α > 1 ? (d.r * d.β)/(d.α - 1) : Inf

function Statistics.var(d::BetaNegBinomial)
    r, α, β = d.r, d.α, d.β
    if α > 2
        numerator = r * β * (r + α - 1) * (β + α - 1)
        denominator = (α - 2) * (α - 1)^2
        return numerator / denominator
    else
        return Inf
    end
end

function StatsBase.skewness(d::BetaNegBinomial)
    r, α, β = d.r, d.α, d.β
    if α > 3
        numerator = (2 * r + α - 1) * (2 * β + α - 1)
        inner_term = (r * β * (r + α - 1) * (β + α - 1))/(α - 2)  
        denominator = (α - 3) * sqrt(inner_term)
        return numerator / denominator
    else
        return Inf
    end
end

#### evaluate functions CDF, PDF, logPDF an CF

function Distributions.cdf(d::BetaNegBinomial, x::Real)
    r, α, β = d.r, d.α, d.β
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)
    log_terms = Float64[]  # Inicializa un array vacío para almacenar los términos logarítmicos
    for k in 0:x
        log_pdf = SpecialFunctions.loggamma(r + k) + SpecialFunctions.logbeta(α + r, β + k) - SpecialFunctions.loggamma(k + 1) - SpecialFunctions.loggamma(r) - SpecialFunctions.logbeta(α, β)
        push!(log_terms, log_pdf)  # Almacena los términos logarítmicos en el array
    end
    log_cdf_value = LogExpFunctions.logsumexp(log_terms)  # Utiliza logsumexp para sumar los términos logarítmicos
    return exp(log_cdf_value)
end

function Distributions.pdf(d::BetaNegBinomial, x::Real)
    r, α, β = d.r, d.α, d.β
    if !insupport(d, x)
        return -Inf
    end
    x = round(Int, x)
    
    log_pdf = SpecialFunctions.loggamma(r + x) + SpecialFunctions.logbeta(α + r, β + x) - SpecialFunctions.loggamma(x + 1) - SpecialFunctions.loggamma(r) - SpecialFunctions.logbeta(α, β)
              
    return exp(log_pdf)
end

function Distributions.logpdf(d::BetaNegBinomial, x::Real)
    r, α, β = d.r, d.α, d.β
    if !insupport(d, x)
        return -Inf
    end
    x = round(Int, x)
    
    log_pdf = SpecialFunctions.loggamma(r + x) + SpecialFunctions.logbeta(α + r, β + x) - SpecialFunctions.loggamma(x + 1) - SpecialFunctions.loggamma(r) - SpecialFunctions.logbeta(α, β)
              
    return log_pdf
end

function Distributions.quantile(d::BetaNegBinomial, p::Real)

    if p < 0 || p > 1
        throw(DomainError(p, "p must be between 0 and 1"))
    end

    lo = 0
    hi = 10
    while cdf(d, hi) < p
        hi *= 2
    end
    
    # binary search
    while lo < hi
        mid = div(lo + hi, 2)
        if cdf(d, mid) < p
            lo = mid + 1
        else
            hi = mid
        end
    end
    
    return lo
end

function Distributions.cf(d::BetaNegBinomial, t)
    r, α, β = d.r, d.α, d.β
    gamma_factor = SpecialFunctions.gamma(α + r) / (SpecialFunctions.gamma(α) * SpecialFunctions.gamma(r))
    hypergeo_factor = HypergeometricFunctions._₂F₁(r , β, α + β + r, exp(im * t))
    gamma_factor / hypergeo_factor
end

## sampling 
function Distributions.rand(rng::Distributions.AbstractRNG, d::BetaNegBinomial)
    r, α, β = d.r, d.α, d.β
    p = rand(rng, Distributions.Beta(α,β))
    return rand(rng, Distributions.NegativeBinomial(r, p))
end