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

def main():
	import_custom_nodes()
	with torch.inference_mode():
		loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
		loadimage_76 = loadimage.load_image(image="tag_hot.png")

		primitiveboolean = NODE_CLASS_MAPPINGS["PrimitiveBoolean"]()
		primitiveboolean_182 = primitiveboolean.EXECUTE_NORMALIZED(value=False)

		paddleocrnode = NODE_CLASS_MAPPINGS["PaddleOCRNode"]()
		paddleocrnode_241 = paddleocrnode.run(lang="ch", device="gpu", image=get_value_at_index(loadimage_76, 0))

		filterchinesetext = NODE_CLASS_MAPPINGS["FilterChineseText"]()
		filterchinesetext_242 = filterchinesetext.encode(texts_bboxes=get_value_at_index(paddleocrnode_241, 0), texts_string=get_value_at_index(paddleocrnode_241, 1))

		autotranslatechinatovn = NODE_CLASS_MAPPINGS["AutoTranslateChinaToVN"]()
		autotranslatechinatovn_244 = autotranslatechinatovn.encode(texts_bboxes=get_value_at_index(filterchinesetext_242, 0), texts_string=get_value_at_index(filterchinesetext_242, 1))

		unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
		unetloader_92_106 = unetloader.load_unet(unet_name="flux-2-klein-9b-fp8.safetensors", weight_dtype="default")

		createimagebytextnode = NODE_CLASS_MAPPINGS["CreateImageByTextNode"]()
		createimagebytextnode_243 = createimagebytextnode.run(image=get_value_at_index(loadimage_76, 0), bboxes=get_value_at_index(autotranslatechinatovn_244, 0), texts_string=get_value_at_index(autotranslatechinatovn_244, 1))

		getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
		getimagesize_177_11 = getimagesize.execute(image=get_value_at_index(loadimage_76, 0))

		cm_inttofloat = NODE_CLASS_MAPPINGS["CM_IntToFloat"]()
		cm_inttofloat_177_18 = cm_inttofloat.op(a=get_value_at_index(getimagesize_177_11, 0))

		cm_floatbinaryoperation = NODE_CLASS_MAPPINGS["CM_FloatBinaryOperation"]()
		cm_floatbinaryoperation_177_15 = cm_floatbinaryoperation.op(op="Div", a=1024, b=get_value_at_index(cm_inttofloat_177_18, 0))

		cm_inttofloat_177_21 = cm_inttofloat.op(a=get_value_at_index(getimagesize_177_11, 1))

		cm_floatbinaryoperation_177_19 = cm_floatbinaryoperation.op(op="Mul", a=get_value_at_index(cm_floatbinaryoperation_177_15, 0), b=get_value_at_index(cm_inttofloat_177_21, 0))

		cm_floattoint = NODE_CLASS_MAPPINGS["CM_FloatToInt"]()
		cm_floattoint_177_20 = cm_floattoint.op(a=get_value_at_index(cm_floatbinaryoperation_177_19, 0))

		imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
		imageresizekjv2_177_12 = imageresizekjv2.resize(width=1024, height=get_value_at_index(cm_floattoint_177_20, 0), upscale_method="nearest-exact", keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center", divisible_by=2, device="cpu", image=get_value_at_index(loadimage_76, 0), mask=get_value_at_index(loadimage_76, 1), unique_id=12403509139445845994)

		imageresizekjv2_125 = imageresizekjv2.resize(width=get_value_at_index(imageresizekjv2_177_12, 1), height=get_value_at_index(imageresizekjv2_177_12, 2), upscale_method="nearest-exact", keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center", divisible_by=2, device="cpu", image=get_value_at_index(createimagebytextnode_243, 0), unique_id=3597060319182616064)

		comfyswitchnode = NODE_CLASS_MAPPINGS["ComfySwitchNode"]()
		comfyswitchnode_178 = comfyswitchnode.EXECUTE_NORMALIZED(switch=get_value_at_index(primitiveboolean_182, 0), on_false=get_value_at_index(createimagebytextnode_243, 0), on_true=get_value_at_index(imageresizekjv2_125, 0))

		imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
		imagescaletototalpixels_92_85 = imagescaletototalpixels.EXECUTE_NORMALIZED(upscale_method="nearest-exact", megapixels=1, resolution_steps=1, image=get_value_at_index(comfyswitchnode_178, 0))

		vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
		vaeloader_92_107 = vaeloader.load_vae(vae_name="flux2-vae.safetensors")

		vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
		vaeencode_92_84_120 = vaeencode.encode(pixels=get_value_at_index(imagescaletototalpixels_92_85, 0), vae=get_value_at_index(vaeloader_92_107, 0))

		cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
		cliploader_92_111 = cliploader.load_clip(clip_name="qwen_3_8b_fp8mixed.safetensors", type="flux2", device="default")

		cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
		cliptextencode_92_113 = cliptextencode.encode(
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
            clip=get_value_at_index(cliploader_92_111, 0)
        )

		cliptextencode_92_87 = cliptextencode.encode(text="", clip=get_value_at_index(cliploader_92_111, 0))

		comfyswitchnode_181 = comfyswitchnode.EXECUTE_NORMALIZED(switch=get_value_at_index(primitiveboolean_182, 0), on_false=get_value_at_index(loadimage_76, 0), on_true=get_value_at_index(imageresizekjv2_177_12, 0))

		imagescaletototalpixels_92_110 = imagescaletototalpixels.EXECUTE_NORMALIZED(upscale_method="nearest-exact", megapixels=1, resolution_steps=1, image=get_value_at_index(comfyswitchnode_181, 0))

		vaeencode_92_112_117 = vaeencode.encode(pixels=get_value_at_index(imagescaletototalpixels_92_110, 0), vae=get_value_at_index(vaeloader_92_107, 0))

		randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
		randomnoise_92_105 = randomnoise.EXECUTE_NORMALIZED(noise_seed=random.randint(1, 2**64))

		ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
		ksamplerselect_92_102 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")

		joinimagewithalpha = NODE_CLASS_MAPPINGS["JoinImageWithAlpha"]()
		referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
		cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
		getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
		flux2scheduler = NODE_CLASS_MAPPINGS["Flux2Scheduler"]()
		emptyflux2latentimage = NODE_CLASS_MAPPINGS["EmptyFlux2LatentImage"]()
		samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
		vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
		colormatch = NODE_CLASS_MAPPINGS["ColorMatch"]()
		saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

		for q in range(1):
			joinimagewithalpha_141 = joinimagewithalpha.EXECUTE_NORMALIZED(image=get_value_at_index(loadimage_76, 0), alpha=get_value_at_index(loadimage_76, 1))

			referencelatent_92_112_118 = referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(cliptextencode_92_113, 0), latent=get_value_at_index(vaeencode_92_112_117, 0))

			referencelatent_92_84_121 = referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(referencelatent_92_112_118, 0), latent=get_value_at_index(vaeencode_92_84_120, 0))

			referencelatent_92_112_116 = referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(cliptextencode_92_87, 0), latent=get_value_at_index(vaeencode_92_112_117, 0))

			referencelatent_92_84_119 = referencelatent.EXECUTE_NORMALIZED(conditioning=get_value_at_index(referencelatent_92_112_116, 0), latent=get_value_at_index(vaeencode_92_84_120, 0))

			cfgguider_92_114 = cfgguider.EXECUTE_NORMALIZED(cfg=5, model=get_value_at_index(unetloader_92_106, 0), positive=get_value_at_index(referencelatent_92_84_121, 0), negative=get_value_at_index(referencelatent_92_84_119, 0))

			getimagesize_92_108 = getimagesize.EXECUTE_NORMALIZED(image=get_value_at_index(imagescaletototalpixels_92_110, 0), unique_id=4325781458001510556)

			flux2scheduler_92_115 = flux2scheduler.EXECUTE_NORMALIZED(steps=20, width=get_value_at_index(getimagesize_92_108, 0), height=get_value_at_index(getimagesize_92_108, 1))

			emptyflux2latentimage_92_109 = emptyflux2latentimage.EXECUTE_NORMALIZED(width=get_value_at_index(getimagesize_92_108, 0), height=get_value_at_index(getimagesize_92_108, 1), batch_size=1)

			samplercustomadvanced_92_103 = samplercustomadvanced.EXECUTE_NORMALIZED(noise=get_value_at_index(randomnoise_92_105, 0), guider=get_value_at_index(cfgguider_92_114, 0), sampler=get_value_at_index(ksamplerselect_92_102, 0), sigmas=get_value_at_index(flux2scheduler_92_115, 0), latent_image=get_value_at_index(emptyflux2latentimage_92_109, 0))

			vaedecode_92_104 = vaedecode.decode(samples=get_value_at_index(samplercustomadvanced_92_103, 0), vae=get_value_at_index(vaeloader_92_107, 0))

			imageresizekjv2_129 = imageresizekjv2.resize(width=get_value_at_index(getimagesize_177_11, 0), height=get_value_at_index(getimagesize_177_11, 1), upscale_method="nearest-exact", keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center", divisible_by=2, device="cpu", image=get_value_at_index(vaedecode_92_104, 0), unique_id=1642360321667796631)

			colormatch_179 = colormatch.match_color(method="mkl", image_ref=get_value_at_index(joinimagewithalpha_141, 0), image_target=get_value_at_index(imageresizekjv2_129, 0))

			saveimage_94 = saveimage.save_images(filename_prefix="Flux2-Klein-4b-base", images=get_value_at_index(colormatch_179, 0))

			joinimagewithalpha_191 = joinimagewithalpha.EXECUTE_NORMALIZED(image=get_value_at_index(colormatch_179, 0), alpha=get_value_at_index(loadimage_76, 1))

			saveimage_240 = saveimage.save_images(filename_prefix="ComfyUI", images=get_value_at_index(joinimagewithalpha_191, 0))


if __name__ == "__main__":
	main()