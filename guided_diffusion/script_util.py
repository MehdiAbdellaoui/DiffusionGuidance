from .unet import EncoderUNetModel

NUM_CLASSES = 1000
def create_classifier(
    image_size,
    classifier_in_channels,
    classifier_out_channels,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_pool,
    conditioned=False
):

    if conditioned:
        num_classes = NUM_CLASSES
    else:
        num_classes = None

    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 4)
    elif image_size == 8:
        channel_mult = (1,)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=classifier_in_channels,
        model_channels=classifier_width,
        out_channels=classifier_out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        pool=classifier_pool,
        num_classes=num_classes,
        use_fp16=False,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True
    )
