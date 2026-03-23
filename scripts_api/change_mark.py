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
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

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
    # server_instance = server.PromptServer(loop)
    # execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())

from nodes import LoadImage, SaveImage,NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_5 = loadimage.load_image(image="7c5503453c2e55bda4501b0a2b697ce01f06c896d7f92dc89ba7022a71ac5f89.png [input]")

        loadimage_82 = loadimage.load_image(image="4514eecafdc18af8837ea72d1c4de8662a5b5262a0928bce0032c04d142dd1e3.png")

        primitiveint = NODE_CLASS_MAPPINGS["PrimitiveInt"]()
        primitiveint_171_160 = primitiveint.EXECUTE_NORMALIZED(value=63)

        primitiveint_171_161 = primitiveint.EXECUTE_NORMALIZED(value=59)

        primitiveint_171_162 = primitiveint.EXECUTE_NORMALIZED(value=27)

        primitiveint_171_167 = primitiveint.EXECUTE_NORMALIZED(value=1)

        maskboundingbox = NODE_CLASS_MAPPINGS["MaskBoundingBox+"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
        maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
        rmbg = NODE_CLASS_MAPPINGS["RMBG"]()
        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        joinimagewithalpha = NODE_CLASS_MAPPINGS["JoinImageWithAlpha"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            maskboundingbox_71 = maskboundingbox.execute(padding=0, blur=0, mask=get_value_at_index(loadimage_5, 1), image_optional=get_value_at_index(loadimage_5, 0))

            imagecompositemasked_185 = imagecompositemasked.EXECUTE_NORMALIZED(x=get_value_at_index(maskboundingbox_71, 2), y=get_value_at_index(maskboundingbox_71, 3), resize_source=False, destination=get_value_at_index(loadimage_82, 0), source=get_value_at_index(maskboundingbox_71, 1))

            solidmask_171_165 = solidmask.EXECUTE_NORMALIZED(value=1, width=get_value_at_index(primitiveint_171_161, 0), height=get_value_at_index(primitiveint_171_162, 0))

            maskcomposite_171_166 = maskcomposite.EXECUTE_NORMALIZED(x=get_value_at_index(primitiveint_171_160, 0), y=get_value_at_index(primitiveint_171_167, 0), operation="or", destination=get_value_at_index(loadimage_82, 1), source=get_value_at_index(solidmask_171_165, 0))

            rmbg_80_56 = rmbg.process_image(model="RMBG-2.0", sensitivity=1, process_res=1024, mask_blur=0, mask_offset=0, invert_output=False, refine_foreground=False, background="Alpha", background_color="#222222", image=get_value_at_index(maskboundingbox_71, 1))

            invertmask_80_76 = invertmask.EXECUTE_NORMALIZED(mask=get_value_at_index(rmbg_80_56, 1))

            maskcomposite_192 = maskcomposite.EXECUTE_NORMALIZED(x=get_value_at_index(maskboundingbox_71, 2), y=get_value_at_index(maskboundingbox_71, 3), operation="and", destination=get_value_at_index(maskcomposite_171_166, 0), source=get_value_at_index(invertmask_80_76, 0))

            joinimagewithalpha_209 = joinimagewithalpha.EXECUTE_NORMALIZED(image=get_value_at_index(imagecompositemasked_185, 0), alpha=get_value_at_index(maskcomposite_192, 0))

            joinimagewithalpha_171_181 = joinimagewithalpha.EXECUTE_NORMALIZED(image=get_value_at_index(loadimage_82, 0), alpha=get_value_at_index(maskcomposite_171_166, 0))

            saveimage_209 = saveimage.save_images(image=get_value_at_index(joinimagewithalpha_209, 0), path="output_masked.png")
            
            saveimage_171_182 = saveimage.save_images(image=get_value_at_index(joinimagewithalpha_171_181, 0), path="output.png")


if __name__ == "__main__":
	main()