import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from ComfyUI.main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from ComfyUI.utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")



add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS
from types import SimpleNamespace
from  ComfyUI.comfy import model_management
_nodes = None


_nodes
def init_nodes(output_dir: str ="./content/output"):
    print("Init model")
    # import_custom_nodes()
    global _nodes

    n = SimpleNamespace()
    
    
    # import_custom_nodes()
    with torch.inference_mode():
       
        n.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        n.primitiveboolean = NODE_CLASS_MAPPINGS["PrimitiveBoolean"]()
        n.paddleocrnode = NODE_CLASS_MAPPINGS["PaddleOCRNode"]()
        n.filterchinesetext = NODE_CLASS_MAPPINGS["FilterChineseText"]()
        n.autotranslatechinatovn = NODE_CLASS_MAPPINGS["AutoTranslateChinaToVN"]()
        n.unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        n.createimagebytextnode = NODE_CLASS_MAPPINGS["CreateImageByTextNode"]()
        n.getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
        n.cm_inttofloat = NODE_CLASS_MAPPINGS["CM_IntToFloat"]()
        n.cm_floatbinaryoperation = NODE_CLASS_MAPPINGS["CM_FloatBinaryOperation"]()
        n.cm_floattoint = NODE_CLASS_MAPPINGS["CM_FloatToInt"]()
        n.imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        n.comfyswitchnode = NODE_CLASS_MAPPINGS["ComfySwitchNode"]()
        n.imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
        n.vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        n.vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        n.cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        n.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        n.randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        n.ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        n.joinimagewithalpha = NODE_CLASS_MAPPINGS["JoinImageWithAlpha"]()
        n.referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
        n.cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        n.flux2scheduler = NODE_CLASS_MAPPINGS["Flux2Scheduler"]()
        n.emptyflux2latentimage = NODE_CLASS_MAPPINGS["EmptyFlux2LatentImage"]()
        n.samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        n.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        n.colormatch = NODE_CLASS_MAPPINGS["ColorMatch"]()
        n.saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
        n.cliploader_92_111 = n.cliploader.load_clip(clip_name="qwen_3_8b_fp8mixed.safetensors", type="flux2", device="default")
        n.unetloader_92_106 = n.unetloader.load_unet(unet_name="flux-2-klein-9b-fp8.safetensors", weight_dtype="default")
        n.vaeloader_92_107 = n.vaeloader.load_vae(vae_name="flux2-vae.safetensors")
        n.cliptextencode_92_113 = n.cliptextencode.encode(
                text="""Use Figure 1 as the base image and layout reference.
    
                Replace ONLY the text content with the text from Figure 2.
    
                Strict requirements:
                - Keep the exact same layout and composition as Figure 1.
                - Preserve the original font family, font size, font weight, and text color.
                - Keep the exact same text positions, spacing, and alignment.
                - Do NOT move, resize, redesign, or restyle any elements.
                - Do NOT change the background, shapes, icons, or graphics.
                - The final image must look identical to Figure 1 except that the text is replaced with the text from Figure 2.
    
                This is a text replacement task, not a redesign.""",
                clip=get_value_at_index(n.cliploader_92_111, 0)
            )
        
        n.cliptextencode_92_87 = n.cliptextencode.encode(text="", clip=get_value_at_index(n.cliploader_92_111, 0))
        n.saveimage.output_dir = output_dir
    _nodes = n


def inference(origin_image_path, is_resize_needed: bool = False):
    n = _nodes
    prefix_ref_alpha = os.path.basename(os.path.splitext(origin_image_path)[0])+"_ref_alpha.png"
    prefix_no_ref_alpha = os.path.basename(os.path.splitext(origin_image_path)[0])+"_no_ref_alpha.png"
    with torch.inference_mode():
        loadimage_76 = n.loadimage.load_image(image=origin_image_path)

        primitiveboolean_182 = n.primitiveboolean.EXECUTE_NORMALIZED(value=is_resize_needed)

        paddleocrnode_241 = n.paddleocrnode.run(lang="ch", device="gpu", image=get_value_at_index(loadimage_76, 0))

        filterchinesetext_242 = n.filterchinesetext.encode(texts_bboxes=get_value_at_index(paddleocrnode_241, 0), texts_string=get_value_at_index(paddleocrnode_241, 1))

        autotranslatechinatovn_244 = n.autotranslatechinatovn.encode(texts_bboxes=get_value_at_index(filterchinesetext_242, 0), texts_string=get_value_at_index(filterchinesetext_242, 1))

        createimagebytextnode_243 = n.createimagebytextnode.run(image=get_value_at_index(loadimage_76, 0), bboxes=get_value_at_index(autotranslatechinatovn_244, 0), texts_string=get_value_at_index(autotranslatechinatovn_244, 1), font_path="arial.ttf")

        getimagesize_177_11 = n.getimagesize.execute(image=get_value_at_index(loadimage_76, 0))

        cm_inttofloat_177_18 = n.cm_inttofloat.op(a=get_value_at_index(getimagesize_177_11, 0))

        cm_floatbinaryoperation_177_15 = n.cm_floatbinaryoperation.op(op="Div", a=1024, b=get_value_at_index(cm_inttofloat_177_18, 0))

        cm_inttofloat_177_21 = n.cm_inttofloat.op(a=get_value_at_index(getimagesize_177_11, 1))

        cm_floatbinaryoperation_177_19 = n.cm_floatbinaryoperation.op(op="Mul", a=get_value_at_index(cm_floatbinaryoperation_177_15, 0), b=get_value_at_index(cm_inttofloat_177_21, 0))

        cm_floattoint_177_20 = n.cm_floattoint.op(a=get_value_at_index(cm_floatbinaryoperation_177_19, 0))

        imageresizekjv2_177_12 = n.imageresizekjv2.resize(width=1024, height=get_value_at_index(cm_floattoint_177_20, 0), upscale_method="nearest-exact", keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center", divisible_by=2, device="cpu", image=get_value_at_index(loadimage_76, 0), mask=get_value_at_index(loadimage_76, 1), unique_id=12403509139445845994)

        imageresizekjv2_125 = n.imageresizekjv2.resize(width=get_value_at_index(imageresizekjv2_177_12, 1), height=get_value_at_index(imageresizekjv2_177_12, 2), upscale_method="nearest-exact", keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center", divisible_by=2, device="cpu", image=get_value_at_index(createimagebytextnode_243, 0), unique_id=3597060319182616064)

        comfyswitchnode_178 = n.comfyswitchnode.EXECUTE_NORMALIZED(switch=get_value_at_index(primitiveboolean_182, 0), on_false=get_value_at_index(createimagebytextnode_243, 0), on_true=get_value_at_index(imageresizekjv2_125, 0))

        imagescaletototalpixels_92_85 = n.imagescaletototalpixels.EXECUTE_NORMALIZED(upscale_method="nearest-exact", megapixels=1, resolution_steps=1, image=get_value_at_index(comfyswitchnode_178, 0))

        vaeencode_92_84_120 = n.vaeencode.encode(pixels=get_value_at_index(imagescaletototalpixels_92_85, 0), vae=get_value_at_index(n.vaeloader_92_107, 0))

        comfyswitchnode_181 = n.comfyswitchnode.EXECUTE_NORMALIZED(switch=get_value_at_index(primitiveboolean_182, 0), on_false=get_value_at_index(loadimage_76, 0), on_true=get_value_at_index(imageresizekjv2_177_12, 0))

        imagescaletototalpixels_92_110 = n.imagescaletototalpixels.EXECUTE_NORMALIZED(upscale_method="nearest-exact", megapixels=1, resolution_steps=1, image=get_value_at_index(comfyswitchnode_181, 0))

        vaeencode_92_112_117 = n.vaeencode.encode(pixels=get_value_at_index(imagescaletototalpixels_92_110, 0), vae=get_value_at_index(n.vaeloader_92_107, 0))

        randomnoise_92_105 = n.randomnoise.EXECUTE_NORMALIZED(noise_seed=random.randint(1, 2**64))

        ksamplerselect_92_102 = n.ksamplerselect.EXECUTE_NORMALIZED(sampler_name="dpmpp_2m_sde")

        for q in range(1):
            joinimagewithalpha_141 = n.joinimagewithalpha.EXECUTE_NORMALIZED(image=get_value_at_index(loadimage_76, 0), alpha=get_value_at_index(loadimage_76, 1))

            referencelatent_92_112_118 = n.referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(n.cliptextencode_92_113, 0), latent=get_value_at_index(vaeencode_92_112_117, 0))

            referencelatent_92_84_121 = n.referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(referencelatent_92_112_118, 0), latent=get_value_at_index(vaeencode_92_84_120, 0))

            referencelatent_92_112_116 = n.referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(n.cliptextencode_92_87, 0), latent=get_value_at_index(vaeencode_92_112_117, 0))

            referencelatent_92_84_119 = n.referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(referencelatent_92_112_116, 0), latent=get_value_at_index(vaeencode_92_84_120, 0))

            cfgguider_92_114 = n.cfgguider.EXECUTE_NORMALIZED(cfg=5, model=get_value_at_index(n.unetloader_92_106, 0), positive=get_value_at_index(referencelatent_92_84_121, 0), negative=get_value_at_index(referencelatent_92_84_119, 0))

            getimagesize_92_108 = n.getimagesize.execute(image=get_value_at_index(imagescaletototalpixels_92_110, 0))

            flux2scheduler_92_115 = n.flux2scheduler.EXECUTE_NORMALIZED(steps=20, width=get_value_at_index(getimagesize_92_108, 0), height=get_value_at_index(getimagesize_92_108, 1))

            emptyflux2latentimage_92_109 = n.emptyflux2latentimage.EXECUTE_NORMALIZED(width=get_value_at_index(getimagesize_92_108, 0), height=get_value_at_index(getimagesize_92_108, 1), batch_size=1)

            samplercustomadvanced_92_103 = n.samplercustomadvanced.EXECUTE_NORMALIZED(noise=get_value_at_index(randomnoise_92_105, 0), guider=get_value_at_index(cfgguider_92_114, 0), sampler=get_value_at_index(ksamplerselect_92_102, 0), sigmas=get_value_at_index(flux2scheduler_92_115, 0), latent_image=get_value_at_index(emptyflux2latentimage_92_109, 0))

            vaedecode_92_104 = n.vaedecode.decode(samples=get_value_at_index(samplercustomadvanced_92_103, 0), vae=get_value_at_index(n.vaeloader_92_107, 0))

            imageresizekjv2_129 = n.imageresizekjv2.resize(width=get_value_at_index(getimagesize_177_11, 0), height=get_value_at_index(getimagesize_177_11, 1), upscale_method="nearest-exact", keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center", divisible_by=2, device="cpu", image=get_value_at_index(vaedecode_92_104, 0), unique_id=1642360321667796631)

            colormatch_179 = n.colormatch.match_color(method="mkl", image_ref=get_value_at_index(joinimagewithalpha_141, 0), image_target=get_value_at_index(imageresizekjv2_129, 0))

            saveimage_94 = n.saveimage.save_images(filename_prefix=prefix_no_ref_alpha, images=get_value_at_index(colormatch_179, 0))

            joinimagewithalpha_191 = n.joinimagewithalpha.EXECUTE_NORMALIZED(image=get_value_at_index(colormatch_179, 0), alpha=get_value_at_index(loadimage_76, 1))

            saveimage_240 = n.saveimage.save_images(filename_prefix=prefix_ref_alpha, images=get_value_at_index(joinimagewithalpha_191, 0))
            
            del referencelatent_92_112_118
            del referencelatent_92_84_121
            del referencelatent_92_112_116
            del referencelatent_92_84_119
            del vaeencode_92_84_120
            del vaeencode_92_112_117
            del samplercustomadvanced_92_103
            del vaedecode_92_104
def main():
    init_nodes()
    inference("tag_hot.png", False)


if __name__ == "__main__":
    main()
