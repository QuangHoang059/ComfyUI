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
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS

import_custom_nodes()

def main(image_original="icon_nd@.png", image_generated="ComfyUI_temp_ttkbc_00001_.png"):
   
    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        
        loadimage_82 = loadimage.load_image(image=image_original)
        loadimage_5 = loadimage.load_image(image=image_generated)
        
        paddleocrnode = NODE_CLASS_MAPPINGS["PaddleOCRNode"]()

        paddleocrnode_230 = paddleocrnode.run(
            lang="ch", device="gpu", image=get_value_at_index(loadimage_82, 0)
        )

        paddleocrnode_243 = paddleocrnode.run(
                lang="ch", device="gpu", image=get_value_at_index(loadimage_5, 0)
            )
        filterchinesetext = NODE_CLASS_MAPPINGS["FilterChineseText"]()
        
        filterchinesetext_244 = filterchinesetext.encode(
            texts_bboxes=get_value_at_index(paddleocrnode_230, 2),
            texts_string=get_value_at_index(paddleocrnode_230, 1),
        )

        detectbboxnode = NODE_CLASS_MAPPINGS["DetectBBoxNode"]()
        
        detectbboxnode_245 = detectbboxnode.run(
                bboxes_target=get_value_at_index(filterchinesetext_244, 0),
                bboxes=get_value_at_index(paddleocrnode_243, 2),
                texts_string=get_value_at_index(paddleocrnode_243, 1),
            )
        
        bboxtoint = NODE_CLASS_MAPPINGS["BboxToInt"]()
        image_crop_location_exact = NODE_CLASS_MAPPINGS["Image Crop Location Exact"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
        maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
        rmbg = NODE_CLASS_MAPPINGS["RMBG"]()
        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        joinimagewithalpha = NODE_CLASS_MAPPINGS["JoinImageWithAlpha"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            mark_target = get_value_at_index(loadimage_82, 1)
            for i in range(len(get_value_at_index(filterchinesetext_244, 0))):

                bboxtoint_247 = bboxtoint.bboxtoint(
                    index=i, bboxes=get_value_at_index(detectbboxnode_245, 0)
                )

                image_crop_location_exact_265_254 = image_crop_location_exact.main(
                    x=get_value_at_index(bboxtoint_247, 0),
                    y=get_value_at_index(bboxtoint_247, 1),
                    width=get_value_at_index(bboxtoint_247, 2),
                    height=get_value_at_index(bboxtoint_247, 3),
                    edge="original",
                    image=get_value_at_index(loadimage_5, 0),
                )

                imagecompositemasked_265_185 = imagecompositemasked.EXECUTE_NORMALIZED(
                    x=get_value_at_index(bboxtoint_247, 0),
                    y=get_value_at_index(bboxtoint_247, 1),
                    resize_source=False,
                    destination=get_value_at_index(loadimage_82, 0),
                    source=get_value_at_index(image_crop_location_exact_265_254, 0),
                )

                bboxtoint_257 = bboxtoint.bboxtoint(
                    index=i, bboxes=get_value_at_index(filterchinesetext_244, 0)
                )

                solidmask_265_222 = solidmask.EXECUTE_NORMALIZED(
                    value=1,
                    width=get_value_at_index(bboxtoint_257, 2),
                    height=get_value_at_index(bboxtoint_257, 3),
                )

                maskcomposite_265_223 = maskcomposite.EXECUTE_NORMALIZED(
                    x=get_value_at_index(bboxtoint_257, 0),
                    y=get_value_at_index(bboxtoint_257, 1),
                    operation="or",
                    destination=mark_target,
                    source=get_value_at_index(solidmask_265_222, 0),
                )

                rmbg_265_219 = rmbg.process_image(
                    model="RMBG-2.0",
                    sensitivity=1,
                    process_res=1024,
                    mask_blur=0,
                    mask_offset=0,
                    invert_output=False,
                    refine_foreground=False,
                    background="Alpha",
                    background_color="#222222",
                    image=get_value_at_index(image_crop_location_exact_265_254, 0),
                )

                invertmask_265_220 = invertmask.EXECUTE_NORMALIZED(
                    mask=get_value_at_index(rmbg_265_219, 1)
                )

                mark_target = maskcomposite.EXECUTE_NORMALIZED(
                    x=get_value_at_index(bboxtoint_247, 0),
                    y=get_value_at_index(bboxtoint_247, 1),
                    operation="and",
                    destination=get_value_at_index(maskcomposite_265_223, 0),
                    source=get_value_at_index(invertmask_265_220, 0),
                )

            joinimagewithalpha_209 = joinimagewithalpha.EXECUTE_NORMALIZED(
                image=get_value_at_index(imagecompositemasked_265_185, 0),
                alpha=get_value_at_index(mark_target, 0),
            )

            saveimage_221 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(joinimagewithalpha_209, 0),
            )

            joinimagewithalpha_224 = joinimagewithalpha.EXECUTE_NORMALIZED(
                image=get_value_at_index(loadimage_82, 0),
                alpha=get_value_at_index(maskcomposite_265_223, 0),
            )

            saveimage_229 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(joinimagewithalpha_224, 0),
            )

            # joinimagewithalpha_263 = joinimagewithalpha.EXECUTE_NORMALIZED(
            #     image=get_value_at_index(imagecompositemasked_265_185, 0),
            #     alpha=get_value_at_index(loadimage_82, 1),
            # )

            # saveimage_264 = saveimage.save_images(
            #     filename_prefix="ComfyUI",
            #     images=get_value_at_index(joinimagewithalpha_263, 0),
            # )


if __name__ == "__main__":
    main()
