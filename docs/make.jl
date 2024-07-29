using ExtraDist
using Documenter

DocMeta.setdocmeta!(ExtraDist, :DocTestSetup, :(using ExtraDist); recursive=true)

makedocs(;
    modules=[ExtraDist],
    authors="Santiago Jimenez Ramos",
    sitename="ExtraDist.jl",
    format=Documenter.HTML(;
        canonical="https://Santymax98.github.io/ExtraDist.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Santymax98/ExtraDist.jl",
    devbranch="master",
)
