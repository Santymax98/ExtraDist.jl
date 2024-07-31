using ExtraDist
using Documenter
using GR
using Distributions

DocMeta.setdocmeta!(ExtraDist, :DocTestSetup, :(using ExtraDist); recursive=true)

makedocs(;
    modules=[ExtraDist],
    authors="Santiago Jimenez Ramos",
    sitename="ExtraDist.jl",
    format=Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical="https://Santymax98.github.io/ExtraDist.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "index.md",
        "starting.md",
        "Distributions.md"
    ],
)

deploydocs(;
    repo="github.com/Santymax98/ExtraDist.jl",
    devbranch="main",
)
