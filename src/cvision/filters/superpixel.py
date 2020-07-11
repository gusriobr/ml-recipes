import pigsty.segmentation.superpixel as spx
import utils.filters.color as fcolor


def apply_spx(img, tag):
    if "NONE" in tag or tag is None:
        pass  # no superpixel color applied
    if "QUICKSHIFT" in tag:
        img = spx.draw_quickshift_segments(img, average_color=True, random_seed=999111, convert2lab=False)
    elif "SLIC0" in tag:
        img = spx.draw_slic_segments(img, average_color=True, slic_zero=True)
    elif "SLIC" in tag:
        img = spx.draw_slic_segments(img, average_color=True, convert2lab=False)
    elif "KMEANS" in tag:
        img = fcolor.quantization_kmeans(img, n_colors=8)
    elif "FELZEN" in tag:  # felzenszwalb
        img = spx.draw_fz_segments(img, average_color=True)
    elif "MEANSHIFT" in tag:
        if "HUE" in tag or "BN" in tag:
            img = fcolor.quantization_meanshift(img, bandwidth=12)  # value obtained experimentally
        else:
            img = fcolor.quantization_meanshift_ocv(img)
    elif "WATERSHED" in tag:  # felzenszwalb
        img = spx.draw_watershed_segments(img, average_color=True)

    return img
