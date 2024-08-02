using Test
using ExtraDistributions
using Distributions
using Random

@testset "Extra Continuous Tests" begin
    rng = MersenneTwister(2024)
    
    @testset "Alpha - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (α = 1.0, β = 1.0),
            (α = 2.0, β = 2.0),
            (α = 3.0, β = 4.0),
            (α = 5.0, β = 1.0),
            (α = rand(rng), β = rand(rng)),
            (α = rand(rng, Uniform(10, 20)), β = rand(rng, Uniform(5, 10)))
        ]

        for (α, β) in params
            d = Alpha(α, β)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)

            @test_throws MethodError mean(d)

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

    @testset "Argus - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (χ = 1.0, c = 1.0),
            (χ = 2.0, c = 2.0),
            (χ = 3.0, c = 4.0),
            (χ = 5.0, c = 1.0),
            (χ = rand(rng), c = rand(rng)),
            (χ = rand(rng, Uniform(10, 20)), c = rand(rng, Uniform(5, 10)))
        ]

        for (χ, c) in params
            d = Argus(χ, c)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.05

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

    @testset "Benini - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (α = 1.0, β = 1.0, σ = 1.0),
            (α = 2.0, β = 2.0, σ = 2.0),
            (α = 3.0, β = 4.0, σ = 5.0),
            (α = 5.0, β = 1.0, σ = 2.0),
            (α = rand(rng), β = rand(rng), σ = rand(rng)),
            (α = rand(rng, Uniform(10, 20)), β = rand(rng, Uniform(5, 10)), σ = rand(rng, Uniform(0,1)))
        ]

        for (α, β, σ) in params
            d = Benini(α, β, σ)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.05

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

    @testset "Benktander_Type1 - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (a = 1.0, b = 0.5),
            (a = 2.0, b = 1.5),
            (a = 3.0, b = 3.0),
            (a = 4.0, b = 4.0),
            (a = 5.0, b = 7.5) # b = a*(a+1)/2
        ]

        for (a, b) in params
            d = Benktander_Type1(a, b)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.05

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

    @testset "Benktander_Type2 - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (a = 1.0, b = rand(rng)),
            (a = 2.0, b = rand(rng)),
            (a = 3.0, b = rand(rng)),
            (a = rand(rng), b = rand(rng)),
            (a = rand(rng, Uniform(10, 20)), b = rand(rng))
        ]

        for (a, b) in params
            d = Benktander_Type2(a, b)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.05

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

    @testset "Bhattacharjee - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (a = 1.0, b = 2.0, σ = 0.5),
            (a = 2.0, b = 3.0, σ = 1.0),
            (a = 0.0, b = 1.0, σ = 0.1),
            (a = -1.0, b = 1.0, σ = 0.2),
            (a = -2.0, b = -1.0, σ = 0.3)
        ]

        for (a, b, σ) in params
            d = Bhattacharjee(a, b, σ)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.05

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

    @testset "BirnbaumSaunders - sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (μ = 0.0, α = 1.0, β = 1.0),
            (μ = 1.0, α = 0.5, β = 2.0),
            (μ = -1.0, α = 2.0, β = 0.5),
            (μ = 2.0, α = 1.5, β = 1.5),
            (μ = -2.0, α = 0.1, β = 0.1)
        ]
        

        for (μ, α, β) in params
            d = BirnbaumSaunders(μ, α, β)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.9

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

    @testset "Bradford - sampling, pdf, logPDF, cdf, quantile" begin
        params = [rand(rng), rand(rng, Uniform(5,10)), 1, 5.5, 12.1]

        for a in params
            d = Bradford(a)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.05

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

    @testset "Burr- sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (c = 1.0, k = 1.0, λ = 1.0),
            (c = 0.5, k = 1.5, λ = 2.5),
            (c = 2.0, k = 3.0, λ = 4.0),
            (c = 1.5, k = 2.5, λ = 3.5),
            (c = 0.1, k = 0.2, λ = 0.3)
        ]

        for (c, k, λ) in params
            d = Burr(c, k, λ)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            if c * k > 1
                @test mean(data) ≈ mean(d) atol=0.05
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

    @testset "Dagum- sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (0.5, 1.5, 2.5),
            (2.0, 3.0, 4.0),
            (1.5, 2.5, 3.5),
            (0.1, 0.2, 0.3)
        ]

        for (a, b, p) in params
            d = Dagum(a, b, p)
            data = rand(rng, d, 10000)

            @test size(data) == (10000,)
            if a > 1
                @test mean(data) ≈ mean(d) atol= 3
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

    @testset "Gompertz- sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (η = 0.5, b = 1.5),
            (η = 2.0, b = 3.0),
            (η = 1.5, b = 2.5),
            (η = 0.1, b = 0.2)
        ]

        for (η, b) in params
            d = Gompertz(η, b)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol= 0.05        

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

    @testset "Lomax- sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (α = 0.5, λ = 1.5),
            (α = 2.0, λ = 3.0),
            (α = 1.5, λ = 2.5),
            (α = 0.1, λ = 0.2)
        ]

        for (α, λ) in params
            d = Lomax(α, λ)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            if α > 1
                @test mean(data) ≈ mean(d) atol= 0.5
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

    @testset "Maxwell - sampling, pdf, logPDF, cdf, quantile" begin
        params = [rand(rng), rand(rng, Uniform(5,10)), 1, 5.5, 12.1]

        for a in params
            d = Maxwell(a)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol=0.1

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

    @testset "Nakagami- sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (m = 0.6, Ω = 1.5),
            (m = 2.0, Ω = 3.0),
            (m = 1.5, Ω = 2.5),
            (m = 3.1, Ω = 0.2),
            (m = rand(rng, Uniform(0.5, 1)), Ω = rand(rng, Uniform(5, 10)))
       ]

        for (m, Ω) in params
            d = Nakagami(m, Ω)
            data = rand(rng, d, 10^4)

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

    @testset "PERT- sampling, pdf, logPDF, cdf, quantile" begin
        params = [
            (a = 0.0, b = 1.0, m = 2.0),
            (a = 1.0, b = 2.0, m = 3.0),
            (a = -1.0, b = 0.0, m = 1.0),
            (a = -2.0, b = -1.0, m = 0.0),
            (a = 5.0, b = 6.0, m = 7.0)
        ]
        

        for (a, b, m) in params
            d = PERT(a, b, m)
            data = rand(rng, d, 10^4)

            @test size(data) == (10000,)
            @test mean(data) ≈ mean(d) atol= 0.05

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
end