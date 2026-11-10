"""Remove contact with artificial source-crop boundaries before seed grouping."""

def artificial_crop_edges(model, box, bbox, image_size):
    if model == 0:
        return []
    x0, y0, x1, y1 = box
    a0, b0, a1, b1 = bbox
    width, height = image_size
    edges = []
    if x0 > 0 and a0 == x0: edges.append('left')
    if y0 > 0 and b0 == y0: edges.append('top')
    if x1 < width and a1 == x1: edges.append('right')
    if y1 < height and b1 == y1: edges.append('bottom')
    return edges

def has_crop_contact(aliases):
    return any(a['touches_crop'] for a in aliases)
