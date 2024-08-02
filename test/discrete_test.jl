using Test
using ExtraDistributions
using Distributions
using Random

@testset "Extra Discrete Tests" begin
    rng = MersenneTwister(2024)

    @testset "BetaNegBinomial - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
        (r = 1, α = 1.0, β = 2.0),
        (r = 2, α = 0.5, β = 1.5),
        (r = 5, α = 2.0, β = 3.0),
        (r = 10, α = 1.5, β = 0.5),
        (r = 22, α = 0.5, β = 1.1)
        ]


        for (r, α, β) in params
            d = BetaNegBinomial(r, α, β)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            if α > 1
                @test mean(data) ≈ mean(d) atol= 0.9
            else
                @test isinf(mean(d))
            end

            for i in 1:100

                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "Borel - sampling, pdf, logPDF, cdf, quantile" begin
        params = [0.1, 0.25, 0.5, 0.75, 0.9]
        for a in params
            d = Borel(a)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol= 0.1

            for i in 1:100

                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "Conway - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (λ = 0.1, ν = 0.0),
            (λ = 1.0, ν = 0.5),
            (λ = 2.5, ν = 1.0),
            (λ = 5.0, ν = 2.0),
            (λ = 10.0, ν = 3.5)
        ]

        for (λ, ν) in params
            d = Conway(λ, ν)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol= 0.1

            for i in 1:100

                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "Delaporte - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (λ = 0.1, α = 0.5, β = 0.5),
            (λ = 1.0, α = 1.0, β = 0.2),
            (λ = 2.5, α = 2.0, β = 1.0),
            (λ = 5.0, α = 3.0, β = 2.0),
            (λ = 10.0, α = 5.0, β = 4.0)
        ]

        for (λ, α, β) in params
            d = Delaporte(λ, α, β)
            data = rand(rng, d, 10000)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol= 0.5

            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "FlorySchulz - sampling, pdf, logPDF, cdf, quantile" begin
        params = [0.1, rand(rng), rand(rng), rand(rng), 0.9]

        for a in params
            d = FlorySchulz(a)
            data = rand(rng, d, 10000)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol= 1

            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "GaussKuzmin - sampling, pdf, logPDF, cdf, quantile" begin
        d = GaussKuzmin()
        data = rand(rng, d, 10000)
        @test size(data) == (10000,)
        for i in 1:1000
            v = data[i]
            cdf_value = cdf(d, v)
            @test 0.0 <= cdf_value <= 1.0
            @test quantile(d, cdf_value) ≈ v
            pdf_value = pdf(d, v)
            @test 0.0 <= pdf_value
            @test exp(logpdf(d, v)) ≈ pdf_value
        end    
    end
    
    @testset "Logarithmic - sampling, pdf, logPDF, cdf, quantile" begin
        params = [0.1, rand(rng), rand(rng), rand(rng), 0.9]

        for a in params
            d = Logarithmic(a)
            data = rand(rng, d, 10000)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol= 0.09

            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "Rademacher - sampling, pdf, logPDF, cdf, quantile" begin
        d = Rademacher()
        data = rand(rng, d, 10000)
        @test size(data) == (10000,)
        
        @test mean(data) ≈ mean(d) atol= 0.05

        for i in 1:1000
            v = data[i]
            cdf_value = cdf(d, v)
            @test 0.0 <= cdf_value <= 1.0
            @test quantile(d, cdf_value) ≈ v
            pdf_value = pdf(d, v)
            @test 0.0 <= pdf_value
            @test exp(logpdf(d, v)) ≈ pdf_value
        end    
    end

    @testset "Yule - sampling, pdf, logPDF, cdf, quantile" begin
        params = [1, rand(rng), rand(rng, Uniform(0, 5)), rand(rng, Uniform(5, 10)), rand(rng, Uniform(10, 15))]

        for a in params
            d = Yule(a)
            data = rand(rng, d, 10000)

            @test size(data) == (10000,)
            if a > 1
                @test mean(data) ≈ mean(d) atol= 0.05
            else
                @test isnan(mean(d))
            end

            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "Zeta - sampling, pdf, logPDF, cdf, quantile" begin
        params = [1.5, 2.5, rand(rng, Uniform(2, 5)), rand(rng, Uniform(5, 10)), rand(rng, Uniform(10, 15))]

        for a in params
            d = Zeta(a)
            data = rand(rng, d, 10000)

            @test size(data) == (10000,)
            if a > 2
                @test mean(data) ≈ mean(d) atol= 0.9
            else
                @test isnan(mean(d))
            end

            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
                @test 0.0 <= cdf_value <= 1.0
                @test quantile(d, cdf_value) ≈ v
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "ZIB - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (n = 10, θ = 0.3, p = 0.7),
            (n = 5, θ = 0.5, p = 0.5),
            (n = 15, θ = 0.2, p = 0.9),
            (n = 20, θ = rand(rng), p = rand(rng)),
            (n = 25, θ = rand(rng), p = rand(rng))
        ]
    
        for (n, θ, p) in params
            d = ZIB(n, θ, p)
            data = rand(rng, d, 10000)
    
            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.5
    
            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
    
                @test 0.0 <= cdf_value <= 1.0 + 1e-10 
                quantile_value = quantile(d, cdf_value)
                @test quantile_value ≈ v
    
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end
    
    @testset "ZINB - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (r = 10, θ = 0.3, p = 0.7),
            (r = 5, θ = 0.5, p = 0.5),
            (r = 15, θ = 0.2, p = 0.9),
            (r = 20, θ = rand(rng), p = rand(rng)),
            (r = 25, θ = rand(rng), p = rand(rng))
        ]
    
        for (r, θ, p) in params
            d = ZINB(r, θ, p)
            data = rand(rng, d, 10000)
    
            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.5
    
            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
    
                @test 0.0 <= cdf_value <= 1.0
                quantile_value = quantile(d, cdf_value)
                @test quantile_value ≈ v
    
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "ZIP - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (λ = 2.0, p = rand(rng)),
            (λ = 5.0, p = rand(rng)),
            (λ = 1.0, p = rand(rng)),
            (λ = 10.0, p = rand(rng)),
            (λ = 0.5, p = rand(rng))
        ]
    
        for (λ, p) in params
            d = ZIP(λ, p)
            data = rand(rng, d, 10000)
    
            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.06
    
            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
    
                @test 0.0 <= cdf_value <= 1.0
                quantile_value = quantile(d, cdf_value)
                @test quantile_value ≈ v
    
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end

    @testset "Zipf - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (N = 10, s = 1.0),
            (N = 50, s = 1.5),
            (N = 100, s = 2.0),
            (N = 20, s = 0.5),
            (N = 30, s = 3.5)
        ]
    
        for (N, s) in params
            d = Zipf(N, s)
            data = rand(rng, d, 10000)
    
            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.5
    
            for i in 1:100
                v = data[i]
                cdf_value = cdf(d, v)
    
                @test 0.0 <= cdf_value <= 1.0
                quantile_value = quantile(d, cdf_value)
                @test quantile_value ≈ v
    
                pdf_value = pdf(d, v)
                @test 0.0 <= pdf_value
                @test exp(logpdf(d, v)) ≈ pdf_value
            end    
        end
    end
end