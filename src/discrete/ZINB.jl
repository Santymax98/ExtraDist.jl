"""
    ZINB(n, θ, p)

The *Zero-Inflated Negative Binomial (ZINB) distribution* is a discrete probability distribution that combines the negative binomial distribution with an excess of zeros. The probability mass function (PMF) is defined as:

```math
P(X = k) =
\\begin{cases} 
\\theta + (1 - \\theta) \\cdot (1 - p)^r & \\text{if } k = 0, \\
(1 - \\theta) \\cdot \\binom{k + r - 1}{k} p^r (1 - p)^k & \\text{if } k > 0.
\\end{cases}
```

```julia
ZINB()       # equivalent to ZINB(1, 0.5, 0.5)
ZINB(r)      # equivalent to ZINB(r, 0.5, 0.5)
ZINB(r, θ)   # equivalent to ZINB(r, θ, 0.5)

params(d)   # Get the parameters, i.e. (r, θ, p)
```

"""
struct ZINB{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    r::Int
    p::T
    θ::T
    ZINB{T}(r, p, θ) where {T<:Real} = new{T}(r, p, θ)
end

ZINB(r::Integer, θ::Integer, p::Integer; check_args::Bool=true) = ZINB(r, float(θ), float(p); check_args=check_args)
function ZINB(r::Integer, θ::Real, p::Real; check_args::Bool=true)
    @check_args ZINB (r, r >= zero(r)) (θ, zero(θ) <= θ <= one(θ)) (p, zero(p) <= p <= one(p))
    return ZINB{typeof(θ)}(r, θ, p)
end

ZINB() = ZINB{Float64}(1, 0.5, 0.5)
ZINB(r) = ZINB{Float64}(r, 0.5, 0.5)
ZINB(r, θ) = ZIB{Float64}(r, θ, 0.5)

@distr_support ZINB 0 Inf

# parameters
params(d::ZINB) = (d.r, d.θ, d.p)

# statistics
Statistics.mean(d::ZINB) = (1 - d.p) * (1 - d.θ) * d.r / d.θ
Statistics.var(d::ZINB) = (1 - d.p) * d.r / d.θ^2 * (1 - d.θ + d.θ * (1 - d.θ) * d.r * (1 - d.p) / d.θ)

# CDF function
function Distributions.cdf(d::ZINB, x::Real)
    r, p, θ = d.r, d.p, d.θ
    if !insupport(d, x)
        return 0.0
    end
    x = floor(Int, x)
    cdf_value = 0.0
    for k in 0:x
        if k == 0
            cdf_value += p + (1 - p) * θ^r
        else
            log_pdf_k = LogExpFunctions.log1p(-p) + r * log(θ) +  k * LogExpFunctions.log1p(-θ) + SpecialFunctions.logfactorial(k + r - 1) - SpecialFunctions.logfactorial(k) - SpecialFunctions.logfactorial(r - 1) 
            cdf_value += exp(log_pdf_k)
        end
    end
    return cdf_value
end

# PDF function
function Distributions.pdf(d::ZINB, x::Real)
    r, p, θ = d.r, d.p, d.θ
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)
    if x == 0
        pdf_value = p + (1 - p) * θ^r
    else
        log_pdf_x = LogExpFunctions.log1p(-p) + r * log(θ) +  x * LogExpFunctions.log1p(-θ) + SpecialFunctions.logfactorial(x + r - 1) - SpecialFunctions.logfactorial(x) - SpecialFunctions.logfactorial(r - 1) 
        pdf_value = exp(log_pdf_x)
    end
    return pdf_value
end

# logPDF function
function Distributions.logpdf(d::ZINB, x::Real)
    r, p, θ = d.r, d.p, d.θ
    if !insupport(d, x)
        return -Inf  # Usar -Inf para log-probabilidad
    end
    x = round(Int, x)
    if x == 0
        log_pdf_x = log(p + (1 - p) * θ^r)
    else
        log_pdf_x = LogExpFunctions.log1p(-p) + r * log(θ) +  x * LogExpFunctions.log1p(-θ) + SpecialFunctions.logfactorial(x + r - 1) - SpecialFunctions.logfactorial(x) - SpecialFunctions.logfactorial(r - 1) 
    end
    return log_pdf_x
end

# Quantile function using binary search
function Distributions.quantile(d::ZINB, p::Real)
    @assert 0.0 <= p <= 1.0 "p must be between 0 and 1"
    
    r, θ, p_zero = d.r, d.θ, d.p
    
    lo, hi = 0, 100 
    
    while Distributions.cdf(d, hi) < p
        lo = hi
        hi *= 2
    end
    
    while lo < hi
        mid = (lo + hi) ÷ 2
        if Distributions.cdf(d, mid) < p
            lo = mid + 1
        else
            hi = mid
        end
    end
    
    return lo
end

# Sampling function
function Distributions.rand(rng::Distributions.AbstractRNG, d::ZINB)
    r, p, θ = d.r, d.p, d.θ
    u = rand(rng)
    if u < p
        return 0
    else
        return rand(rng, Distributions.NegativeBinomial(r, θ))
    end
end