// Minimal forward-ABI gate for the RimDroid ARM64 Mono experiment.
// Keep this probe small: Unity's full 286-symbol surface is a later gate.
GO(mono_set_dirs, vFpp)
GO(mono_set_assemblies_path, vFp)
GO(mono_config_parse, vFp)
GO(mono_jit_init_version, pFpp)
GO(mono_jit_cleanup, vFp)
GO(mono_domain_assembly_open, pFpp)
GO(mono_assembly_get_image, pFp)
GO(mono_class_from_name, pFppp)
GO(mono_class_get_method_from_name, pFpi)
GO(mono_runtime_invoke, pFpppp)
GO(mono_object_unbox, pFp)
GO(mono_get_runtime_build_info, pFv)
