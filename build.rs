use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use shaderc;

fn build_shaders() -> Result<(), Box<dyn Error>> {
    const SHADER_SOURCE_DIRECTORY: &'static str = "resources/shaders";

    let mut out_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    out_dir.push("resources.out");
    out_dir.push("shaders");

    fs::create_dir_all(&out_dir)?;

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_source_language(shaderc::SourceLanguage::GLSL);

    for entry in fs::read_dir(&SHADER_SOURCE_DIRECTORY)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().unwrap().to_str().unwrap();
        let extension = path.extension().unwrap();

        println!("cargo:rerun-if-changed={}", path.display());

        let shader_source = fs::read_to_string(&path)?;
        let shader_kind = match extension.to_str().unwrap() {
            "ps" => shaderc::ShaderKind::Fragment,
            "vs" => shaderc::ShaderKind::Vertex,
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "compute" => shaderc::ShaderKind::Compute,
            unknown => {
                println!("cargo:warning=Couldn't guess shader kind for the shader {}: extension {} isn't known. Will try to infer from source.", path.display(), unknown);
                shaderc::ShaderKind::InferFromSource
            }
        };

        let binary_result = compiler.compile_into_spirv(
            &shader_source,
            shader_kind,
            name,
            "main",
            Some(&options),
        )?;

        let mut out_extension = extension.to_owned();
        out_extension.push(".spv");

        let mut out_path = out_dir.clone();
        out_path.push(name);
        out_path.set_extension(&out_extension);

        fs::write(&out_path, binary_result.as_binary_u8())?;
    }

    Ok(())
}

fn copy_resources_folder(path: &Path) -> Result<(), Box<dyn Error>> {
    let resources_root = Path::new("resources").join(path);
    for dir_entry in fs::read_dir(&resources_root)? {
        let dir_entry = dir_entry?;
        if dir_entry.file_type()?.is_file() {
            let new_path = Path::new("resources.out")
                .join(path)
                .join(dir_entry.path().strip_prefix(&resources_root)?);

            fs::create_dir_all(new_path.parent().unwrap())?;
            fs::copy(dir_entry.path(), new_path)?;
            println!("cargo:rerun-if-changed={}", dir_entry.path().display());
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    build_shaders()?;
    copy_resources_folder(Path::new("textures"))?;
    Ok(())
}
