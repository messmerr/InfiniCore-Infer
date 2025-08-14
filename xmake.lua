local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

target("infinicore_infer")
    set_kind("shared")

    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")
    set_optimize("fastest")

    -- Optional feature toggles via environment
    if os.getenv("INFINICCL_HAS_RS_AG") == "1" then
        add_defines("INFINICCL_HAS_RS_AG")
    end
    if os.getenv("INFINIRT_HAS_EVENT") == "1" then
        add_defines("INFINIRT_HAS_EVENT")
    end

    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()
