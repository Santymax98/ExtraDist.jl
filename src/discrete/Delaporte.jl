"""
    Delaporte(λ,α,β)

 
A *Delaporte* distribution is a discrete probability distribution that can be viewed as a compound distribution. It combines a Poisson distribution (with mean `λ`) and a Gamma distribution (with shape parameters `α` and `β`). The probability mass function (PMF) of the Delaporte distribution is given by:

```math
P(X = k) = \\sum_{i=0}^{k} \\frac{\\Gamma(\\alpha + i) \\beta^i \\lambda^{k-i} e^{-\\lambda}}{\\Gamma(\\alpha) i! (k-i)!}, \\quad k \\in \\{0, 1, 2, \\dots\\}
```

```julia
Delaporte()        # equivalent to Delaporte(1, 1, 1)
Delaporte(λ)       # equivalent to Delaporte(λ, 1, 1)
Delaporte(λ, α)    # equivalent to Delaporte(r, α, α)

params(d)        # Get the parameters, i.e. (λ, α, β)
```

External link:

* [Delaporte distribution on Wikipedia](https://en.wikipedia.org/wiki/Delaporte_distribution)
"""
struct Delaporte{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    λ::T
    α::T
    β::T
    Delaporte{T}(λ, α, β) where {T<:Real} = new{T}(λ, α, β)
end

Delaporte(λ::Integer, α::Integer, β::Integer; check_args::Bool=true) = Delaporte(float(λ), float(α), float(β); check_args=check_args)

function Delaporte(λ::Real, α::Real, β::Real; check_args::Bool=true)
    @check_args Delaporte (λ, λ > zero(λ)) (α, zero(α) < α) (β, zero(β) < β)
    return Delaporte{typeof(λ)}(λ, α, β)
end

Delaporte() = Delaporte{Float64}(1.0, 1.0, 1.0)
Delaporte(λ) = Delaporte{Float64}(λ, 1.0, 1.0)
Delaporte(λ, α) = Delaporte{Float64}(λ, α, α)

@distr_support Delaporte 0 Inf


# parameters

params(d::Delaporte) = (d.λ, d.α, d.β)
partype(::Delaporte{T}) where {T} = T

#Statistic

Statistics.mean(d::Delaporte) = d.λ + (d.α * d.β)

Statistics.var(d::Delaporte) = d.λ + (d.α * d.β)*(1 +d.β)

function StatsBase.mode(d::Delaporte)
    λ, α, β = d.λ, d.α, d.β
    z = (α - 1) * β + λ
    
    if z == floor(z)
        return [z, z + 1]
    else
        return floor(z)
    end
end


#### evaluate functions CDF, PDF, logPDF an CF

function Distributions.cdf(d::Delaporte, x::Real)
    λ, α, β = d.λ, d.α, d.β
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)
    cdf_value = 0.0
    for j in 0:x
        pdf_value = 0.0 
        for i in 0:j
            num = exp(SpecialFunctions.loggamma(α + i) + i * log(β) + (j - i) * log(λ))
            den = exp(SpecialFunctions.loggamma(α) + SpecialFunctions.logfactorial(i) + (α + i) * log(1 + β) + SpecialFunctions.logfactorial(j - i))
            term = num / den
            pdf_value += term
        end
        cdf_value += pdf_value
    end
    return cdf_value * exp(-λ)
end

function Distributions.pdf(d::Delaporte, x::Real)
    λ, α, β = d.λ, d.α, d.β
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)
    log_pdf_value = 0.0  # iniciamos con exp(-λ)
    for i in 0:x
        log_num = SpecialFunctions.loggamma(α + i) + i*log(β) + (x - i)*log(λ)
        log_den = SpecialFunctions.loggamma(α) + SpecialFunctions.logfactorial(i) + (α + i)*log(1 + β) + SpecialFunctions.logfactorial(x - i)
        log_term = log_num - log_den
        log_pdf_value += log_term 
    end
    return exp(log_pdf_value - λ)
end

function Distributions.logpdf(d::Delaporte, x::Real)
    λ, α, β = d.λ, d.α, d.β
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)
    log_pdf_value = 0  # iniciamos con exp(-λ)
    for i in 0:x
        log_num = SpecialFunctions.loggamma(α + i) + i*log(β) + (x - i)*log(λ)
        log_den = SpecialFunctions.loggamma(α) + SpecialFunctions.logfactorial(i) + (α + i)*log(1 + β) + SpecialFunctions.logfactorial(x - i)
        log_term = log_num - log_den
        log_pdf_value += log_term 
    end
    return log_pdf_value - λ
end

function Distributions.quantile(d::Delaporte, p::Real)

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

## sampling 
function Distributions.rand(rng::Distributions.AbstractRNG, d::Delaporte)
    λ, α, β = d.λ, d.α, d.β
    
    Z = rand(rng, Distributions.Gamma(α, β))
    Y = rand(rng, Distributions.Poisson(Z))
    X = rand(rng, Distributions.Poisson(λ))
    
    return Y + X
end