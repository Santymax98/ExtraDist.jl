using ExtraDistributions
using Documenter
using GR
using Distributions

DocMeta.setdocmeta!(ExtraDistributions, :DocTestSetup, :(using ExtraDistributions); recursive=true)

makedocs(;
    modules=[ExtraDistributions],
    authors="Santiago Jimenez Ramos",
    sitename="ExtraDistributions.jl",
    format=Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical="https://Santymax98.github.io/ExtraDistributions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "index.md",
        "starting.md",
        "Compatibility.md",
        "Distributions.md"
    ],
)

deploydocs(;
    repo="github.com/Santymax98/ExtraDistributions.jl",
    devbranch="main",
)
