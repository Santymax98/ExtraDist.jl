"""
    CrystalBall(α, m, x̄, σ)

A *CrystalBall* distribution is commonly used to model various lossy processes in high-energy physics. The probability density function (PDF) of the Crystal Ball distribution is defined as:

```math
f(x; \\alpha, n, \\bar{x}, \\sigma) =
\\begin{cases} 
N \\cdot \\exp\\left(-\\frac{(x - \\bar{x})^2}{2\\sigma^2}\\right), & \\text{for } \\frac{x - \\bar{x}}{\\sigma} > -\\alpha \\
N \\cdot A \\cdot \\left(B - \\frac{x - \\bar{x}}{\\sigma}\\right)^{-n}, & \\text{for } \\frac{x - \\bar{x}}{\\sigma} ≤ -\\alpha
\\end{cases}
```
where:
* ``A = \\left(\\frac{n}{|\\alpha|}\\right)^n \\cdot \\exp\\left(-\\frac{|\\alpha|^2}{2}\\right)``
* ``B = \\frac{n}{|\\alpha|} - |\\alpha|``
* ``N = \\frac{1}{\\sigma(C + D)}``
* ``C = \\frac{n}{|\\alpha|} \\cdot \\frac{1}{n - 1} \\cdot \\exp\\left(-\\frac{|\\alpha|^2}{2}\\right)``
* ``D = \\sqrt{\\frac{\\pi}{2}} \\left(1 + \\text{erf}\\left(\\frac{|\\alpha|}{\\sqrt{2}}\\right)\\right)``

```julia
CrystalBall(α, m)        # equivalent to CrystalBall(α, m, 0, 1)
CrystalBall(α, m, x̄)     # equivalent to CrystalBall(α, m, x̄, 1)

params(d)        # Get the parameters, i.e. (α, m, x̄, σ)
```

External links:

* [CrystalBall distribution on Wikipedia](https://en.wikipedia.org/wiki/Crystal_Ball_function)
"""
struct CrystalBall{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    α::T # shape
    m::T # shape
    x̄::T # location
    σ::T # scale 
    CrystalBall{T}(α, m, x̄, σ) where {T<:Real} = new{T}(α, m, x̄, σ)
end

# Constructor for integer inputs
CrystalBall(α::Integer, m::Integer, x̄::Integer, σ::Integer; check_args::Bool=true) = CrystalBall(float(α), float(m), float(x̄), float(σ); check_args=check_args)

# Main constructor with argument checking
function CrystalBall(α::Real, m::Real, x̄::Real, σ::Real; check_args::Bool=true)
    @check_args CrystalBall (α, α > zero(α)) (m, m > one(m)) (σ, σ > zero(σ))
    return CrystalBall{typeof(α)}(α, m, x̄, σ)
end

CrystalBall(α, m) = CrystalBall{Float64}(α, m, 0, 1)
CrystalBall(α, m, x̄) = CrystalBall{Float64}(α, m, x̄, 1)

@distr_support CrystalBall -Inf Inf

# Parameters
params(d::CrystalBall) = (d.α, d.m, d.x̄, d.σ)
@inline partype(d::CrystalBall{T}) where {T<:Real} = T

Base.eltype(::Type{CrystalBall{T}}) where {T} = T
#Location, scale and shape
location(d::CrystalBall) = d.x̄
shape(d::CrystalBall) = (d.α, d.m)
scale(d::CrystalBall) = d.σ

A(d::CrystalBall) = (d.m/abs(d.α))^d.m * exp(-abs(d.α)^2 / 2)
B(d::CrystalBall) = d.m/abs(d.α) - abs(d.α)
C(d::CrystalBall) = d.m/abs(d.α) * 1/(d.m - 1) * exp(-abs(d.α)^2 / 2)
D(d::CrystalBall) = sqrt(π/2) * (1 + SpecialFunctions.erf(abs(d.σ)/sqrt(2)))
N(d::CrystalBall) = 1/(d.σ * (C(d) + D(d)))

# (mean and variance)
function moments(d::CrystalBall, k::Integer)
    α, m, x̄, σ = d.α, d.m, d.x̄, d.σ
    n = N(d)
    # gaussian integrate
    bound = x̄ - α * σ
    normal_part, _ = QuadGK.quadgk(x -> x^k * exp(-1/2 * ((x - x̄)/σ)^2), bound, Inf)
    
    # LAW POW integrate
    power_part, _ = QuadGK.quadgk(x -> x^k * A(d) * (B(d) - (x - x̄)/σ)^(-m), -Inf, bound)
    
    return n * (normal_part + power_part)
end

Statistics.mean(d::CrystalBall) = moments(d, 1)
Statistics.var(d::CrystalBall) = moments(d, 2) - moments(d, 1)^2

# Evaluate functions CDF, PDF, logPDF, quantile
function Distributions.cdf(d::CrystalBall, x::Real)
    α, m, x̄, σ = d.α, d.m, d.x̄, d.σ
    z = (x - x̄)/σ
    n = N(d)
    _insupport = insupport(d, x)
    if _insupport
        if z > -α
            return n * σ * (sqrt(π/2) * (SpecialFunctions.erf(z/sqrt(2)) + SpecialFunctions.erf(α/sqrt(2))) + m * exp(-α^2 / 2))/(α * (m - 1))
        else
            return n * A(d) * σ * (B(d) - z)^(1-m) /(m - 1)
        end
    else
        return 0.0
    end 
end

function Distributions.pdf(d::CrystalBall, x::Real)
    α, m, x̄, σ = d.α, d.m, d.x̄, d.σ
    z = (x - x̄)/σ
    n = N(d)
    _insupport = insupport(d, x)
    if _insupport
        if z > -α
            return n * exp(-1/2 * z^2)
        else
            return n * A(d) * (B(d) - z)^(-m)
        end
    else
        return 0.0
    end 
end

function Distributions.logpdf(d::CrystalBall, x::Real)
    α, m, x̄, σ = d.α, d.m, d.x̄, d.σ
    z = (x - x̄)/σ
    n = N(d)
    _insupport = insupport(d, x)
    if _insupport
        if z > -α
            return log(n) - 1/2 * z^2
        else
            return log(n) + log(A(d)) -m * log(B(d) - z)
        end
    else
        return 0.0
    end 
end

function Distributions.quantile(d::CrystalBall, p::Real)
    α, m, x̄, σ = d.α, d.m, d.x̄, d.σ
    cdf_func(x) = Distributions.cdf(d, x) - p
    lower_bound = x̄ - 1e6 * σ
    upper_bound = x̄ + 1e6 * σ
    x = Roots.find_zero(cdf_func, [lower_bound, upper_bound], Roots.Brent())
    return x
end

function Distributions.rand(rng::Distributions.AbstractRNG, d::CrystalBall)
    α, m, x̄, σ = d.α, d.m, d.x̄, d.σ
    while true
        # Generar candidato de una distribución normal
        z = rand(rng, normal_dist)
        if z > -α
            # Aceptar si está en la parte gaussiana
            return x̄ + σ * z
        else
            # Generar candidato de la cola de ley de potencia
            u = rand(rng)
            candidate = x̄ + σ * (B(d) - u^(-1/(m-1)))
            if rand(rng) < exp(-0.5 * (z^2 - ((B(d) - (candidate - x̄)/σ)^2)))
                return candidate
            end
        end
    end
end