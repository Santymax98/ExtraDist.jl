"""
    Rademacher()

A *Rademacher distribution* is a discrete probability distribution where a random variate ``X`` has a ``50\\%`` chance of being ``+1`` and a ``50\\%`` chance of being ``-1``.

```math
P(X = k) = \\begin{cases}
0.5 & \\quad \\text{for } k = -1, \\\\
0.5 & \\quad \\text{for } k = +1.
\\end{cases}
```

```julia
Rademacher()    # Rademacher distribution 
```

External link:

* [Rademacher distribution on Wikipedia](https://en.wikipedia.org/wiki/Rademacher_distribution)
"""
struct Rademacher <: Distributions.DiscreteUnivariateDistribution
end

@distr_support Rademacher -1 1

# params
params(d::Rademacher) = (println("non-params"))
# statistics 
Statistics.mean(d::Rademacher) = 0
Statistics.var(d::Rademacher) = 1
Statistics.median(d::Rademacher) = 0
StatsBase.mode(d::Rademacher) = NaN
StatsBase.skewness(d::Rademacher) = 0
StatsBase.kurtosis(d::Rademacher) = -2
StatsBase.entropy(d::Rademacher) = log(2)

#evaluate functions CDF, PDF, logPDF, MGF, CF, Quantil
function Distributions.cdf(d::Rademacher, x::Real) 
    if !insupport(d, x)
        return 0.0
    end
    x = round(Int, x)
    if x < 1
        return 0.5
    else
        return 1.0
    end
end
# pdf 
function Distributions.pdf(d::Rademacher, x::Real)
    if !insupport(d, x)
        return 0.0
    end
    x = floor(x)
    if x == 0
        println("zero does not belong to Rademacher distribution support")
        return 0.0
    else
        return 0.5
    end
end

function Distributions.logpdf(d::Rademacher, x::Real)
    if !insupport(d, x)
        return -Inf
    end
    x = floor(x)
    values = [-1, 0, 1]
    if x in values
        return log(0.5)
    else
        return -Inf
    end
end

function Distributions.mgf(d::Rademacher, t)
    return cosh(t)
end

function Distributions.cf(d::Rademacher, t)
    return cos(t)    
end

function Distributions.quantile(d::Rademacher, p::Real)
    if p < 0 || p > 1
        throw(DomainError(p, "p must be in [0, 1]"))
    elseif p <= 0.5
        return -1
    else
        return 1
    end
end

# sampling
function Distributions.rand(rng::Distributions.AbstractRNG, d::Rademacher)
    return rand(rng, Bool) ? 1 : -1
end