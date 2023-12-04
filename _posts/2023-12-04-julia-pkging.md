---
layout: post
title: "Julia packaging at the command line"
description: "A simple Calculator package in Julia Language"
date: 2023-12-04
categories: [Julia, Misc]
comments: true
---

This week, I worked on Julia packaging. Since I knew close to nothing at first, I created a [bash script](https://github.com/thierrymoudiki/scaffold_julia) for generating a **package architecture at the command line** -- that ChatGPT definitely helped me craft, but in case you want to do the same, you'll have to ask precise questions and check its answers. 

There are actually other ways to do that, I found out. For example by using Julia function `generate`. 

I think I'll continue to use and improve the script anyway, because I know exactly what it does at each step. Here it is ([can also be found here](https://github.com/thierrymoudiki/scaffold_julia)), use `Calculator` as package name when asked: 

```bash
#!/bin/bash

# make it an executable 
# chmod +x create_calculator_package.sh
# then run
# ./create_calculator_package.sh

# Prompt user to enter the package name
read -p "Enter package name: " package_name

# Replace spaces with underscores in the package name
package_name="${package_name// /_}"

# Create a folder for the package (folder name will be 'Calculator.jl')
mkdir "${package_name}.jl"
cd "${package_name}.jl" || exit

# Create the src and test directories
mkdir src
mkdir test

# Create necessary files within the src directory
touch src/"${package_name}.jl"
touch src/add.jl
touch src/subtract.jl

# Create necessary files within the test directory
touch test/runtests.jl

# Create Project.toml and populate it with initial information
cat <<EOF > Project.toml
name = "$package_name"
uuid = "$(uuidgen)"
authors = ["Your Name <your@email.com>"]
version = "0.1.0"

[deps]
# Dependencies, if any, will be listed here

[compat]
julia = "1.6"  # Minimum Julia version required by the package
EOF

# Populate files with initial code

# src/Calculator.jl
cat <<EOF > src/"${package_name}.jl"
module $package_name

include("add.jl")
include("subtract.jl")

end # module
EOF

# src/add.jl
cat <<EOF > src/add.jl
function add(a, b)
    return a + b
end
EOF

# src/subtract.jl
cat <<EOF > src/subtract.jl
function subtract(a, b)
    return a - b
end
EOF

# test/runtests.jl
cat <<EOF > test/runtests.jl
using Test
using $package_name

# Test cases for add and subtract functions
@test add(2, 3) == 5
@test subtract(5, 2) == 3

# Additional test cases can be added here

# To run tests, execute this file or include it in a test suite
EOF

# Optionally, create a README.md file
touch README.md

# Output success message
echo "$package_name package structure, Project.toml, and initial code created successfully."
```

**Using the package in Julia console:**

```julia
(@v1.8) pkg> add https://github.com/thierrymoudiki/scaffold_julia/tree/main/Calculator2.jl
     Cloning git-repo `https://github.com/thierrymoudiki/Calculator2.jl.git`
    Updating git-repo `https://github.com/thierrymoudiki/Calculator2.jl.git`
    Updating registry at `~/.julia/registries/General.toml`
   Resolving package versions...
    Updating `~/.julia/environments/v1.8/Project.toml`
  [05fd6518] + Calculator2 v0.1.0 `https://github.com/thierrymoudiki/Calculator2.jl.git#main`
    Updating `~/.julia/environments/v1.8/Manifest.toml`
  [05fd6518] + Calculator2 v0.1.0 `https://github.com/thierrymoudiki/Calculator2.jl.git#main`
Precompiling project...
  1 dependency successfully precompiled in 2 seconds
julia> import Calculator2 as calc
julia> calc.add(1, 2)
3
```


![image-title-here]({{base}}/images/2023-12-04/2023-12-04-image1.png){:class="img-responsive"}