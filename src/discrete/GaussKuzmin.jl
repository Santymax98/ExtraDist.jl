"""
    GaussKuzmin()

An *GaussKuzmin* distribution is defined by the following probability density function (PDF):

```math
f(x) =
```
where:

```julia
GaussKuzmin()        # equivalent to GaussKuzmin()

params(d)        # Get the parameters, i.e. none
```
"""
struct GaussKuzmin <: Distributions.DiscreteUnivariateDistribution
end

@distr_support GaussKuzmin 1 Inf

params(d::GaussKuzmin) = (println("non-params"))

# Accessors for individual parameters

# Some statistics 
Statistics.mean(d::GaussKuzmin) = Inf
Statistics.var(d::GaussKuzmin) = Inf
Statistics.median(d::GaussKuzmin) = 2
StatsBase.mode(d::GaussKuzmin) = 1

# Evaluate functions CDF, PDF, logPDF, quantile
function Distributions.cdf(d::GaussKuzmin, x::Real)
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)
    
    return 1 - log2((x+2)/(x+1))
end

function Distributions.pdf(d::GaussKuzmin, x::Real)
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)

    return -log2(1 - (1/(x+1)^2))
end

function Distributions.logpdf(d::GaussKuzmin, x::Real)
    if !insupport(d, x)
        return -Inf
    end
    x = round(Int, x)
    return log(-log2(1 - (1/(x+1)^2)))
end

function Distributions.quantile(d::GaussKuzmin, p::Real)
    if p < 0.0 || p > 1.0
        throw(ArgumentError("p must be in [0, 1]"))
    end

    return round(Int,(2 - 2^(1-p))/(2^(1-p) - 1))
end

# Sampling
function Distributions.rand(rng::Distributions.AbstractRNG, d::GaussKuzmin)
    u = rand(rng)
    return Distributions.quantile(d, u) 
end