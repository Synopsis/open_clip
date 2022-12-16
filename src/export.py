from evaluation import *
from upyog.all import *
from upyog.cli import Param as P

if True:
    import sys
    sys.path.append("/home/synopsis/git/CinemaNet-Training/")
    sys.path.append("/home/synopsis/git/YOLOX-Custom/")
    sys.path.append("/home/synopsis/git/YOLO-CinemaNet/")
    sys.path.append("/home/synopsis/git/icevision/")
    sys.path.append("/home/synopsis/git/labelling-workflows/")
    sys.path.append("/home/synopsis/git/amalgam/")
    sys.path.append("/home/synopsis/git/cinemanet-multitask-classification/")
    sys.path.append("/home/synopsis/git/Synopsis.py/")


meta = """
Fine-tuned version of OpenAI's ViT-L-14_336.

The model was trained for 1 epoch on 660k cinemantic images (from ShotDeck) that were auto labelled with
1. BLIP
2. CinemaNet teacher models thresholded at 90%
"""


# Helper functions
def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    if isinstance(image_size, tuple):
        image_size = image_size[0]

    if hasattr(model, "context_length"):
          context_length = model.context_length
    else: context_length = model.positional_embedding.shape[0]

    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def create_clip_schema(
    model: torch.nn.Module,
    version: str,
    model_str: str = "",
    meta: str = "",
):
    image_size = model.visual.image_size
    if isinstance(image_size, tuple): image_size = image_size[0]
    assert isinstance(image_size, int)

    if hasattr(model, "context_length"):
          context_length = model.context_length
    else: context_length = model.positional_embedding.shape[0]

    schema = {
        "version": version,
        "model": model_str,
        "licence": "Commercial",
        "meta": meta,
        "input_name": "Image",  # Doesn't matter?
        "output_names": ["clip_image_embedding"],
        "output_embedding_size": [768],
        "text_encoder_context_length": context_length,
        "preprocessing": {
            "input_height": image_size,
            "input_width": image_size,
            "normalisation_mean": list(model.visual.image_mean),
            "normalisation_stdev": list(model.visual.image_std),
            "interpolation": "bicubic",
            "output_format": "RGB",
            "resize_method": "squish",
            "pad_fill": None,
        }
    }

    assert json.dumps(schema), f"Schema is not JSON exportable!"

    return schema


def setup_model(device: torch.device, path_model):
    # 99% of this function is copied over from OpenAI's official CLIP repo: 
    # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L126-L192

    # Use the string representation of the device as this is what OpenAI's code below expects
    device = str(device)

    model = torch.jit.load(path_model, map_location=device)
    model = model.eval()

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    # Takes ~0.1s on M1 CPU
    # start = time.time()
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    # end = time.time()
    # print(f"Patching device the first time took: {end - start} seconds")

    # NOTE: This seems extremely stupid to do again, but it's how OpenAI does it and we know
    # it works so it's a good starting point. We can revisit later.. or not
    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        # Takes ~0.4s on M1 CPU
        # start = time.time()
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()
        # end = time.time()
        # print(f"Patching device the second time took: {end - start} seconds")

    # NOTE: We load the entire CLIP model, but only need the visual embedding, so we discard
    # the text encoder part of the model here
    return model

@call_parse
def export_model(
    # Args
    variant:       P("Model arch", str) = "ViT-L-14-336",
    ckpt_path:     P("Path to the fine-tuned model", str) = None,
    pretrained:    P("Pretrained model tag", str) = "openai",
    alpha:         P("Alpha to blend `pretrained` and `ckpt_path` model", float) = 0.5,
    save_dir:      P("Path to save the model and schema to", str) = None,
    version:       P("Model Version", str) = "1.0.0-RC1",
    jit_export:    P("Do JIT export?", bool) = False,
):
    args = deepcopy(locals())

    # Load model
    # -- Stock model on CPU
    stock = load_model(variant, "cpu", pretrained, None).eval()
    sd_stock = stock.state_dict()

    # -- Main model on GPU
    model = load_model(variant, 0, None, ckpt_path).eval()
    sd_finetune = {k:v.detach().cpu() for k,v in model.state_dict().items()}

    # -- Interpolate weights
    interpolated_wts = interpolate_weights(sd_finetune, sd_stock, alpha)
    model.load_state_dict(interpolated_wts)

    del interpolated_wts, sd_stock, sd_finetune


    # Setup save dir / filenames
    save_dir = Path(save_dir) / version
    save_dir.mkdir(exist_ok=True)
    fname = f"CinemaCLIP_{variant}_{version}"
    save_path_model = str(save_dir / f"{fname}.pt")
    save_path_state_dict = str(save_dir / f"{fname}_weights.pth")
    save_path_schema = str(save_dir / f"CinemaCLIPSchema{version}.json")

    # Make schema
    schema = create_clip_schema(model, version, variant, meta)
    with open(save_path_schema, "w") as f:
        json.dump(schema, f, indent=4)
    logger.success(f"Wrote schema to {save_path_schema}")

    # Save export kwargs
    with open(save_dir / "export_kwargs.json", "w") as f:
        json.dump(args, f, indent=4)

    if not jit_export:
        torch.save(model.state_dict(), save_path_state_dict)

    else:
        # JIT Model
        mjit = trace_model(model, batch_size=2, device="cuda:0")
        torch.jit.save(mjit, save_path_model)
        logger.success(f"Saved model to {save_path_model}")

        del mjit, model


        # Test Inference
        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda:0")

        logger.info(f"Testing models on different devices & batch sizes")
        for device in [cpu_device, gpu_device]:
            m = setup_model(device, save_path_model)
            img_size = (schema["preprocessing"]["input_width"], schema["preprocessing"]["input_height"])

            # Test at different batch sizes
            for batch_size in [1,2]:
                xi = torch.rand((batch_size, 3, *img_size), device=device)
                xt = torch.zeros((batch_size, schema["text_encoder_context_length"]), dtype=torch.int, device=device)
                m.encode_image(xi)
                m.encode_text(xt)

            del m

        logger.success(f"Success!")
