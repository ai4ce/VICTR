"""Classical colour/contour geometry. Coordinates are normalized 640x480 views.

Head images are squashed like the dataset before processing. No learned features.
"""
import cv2
import numpy as np


def normalize(image):
    return cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)


def white_mask(image, head=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rgb = image.astype(float)
    mask = ((hsv[:, :, 1] < (62 if head else 65))
            & (hsv[:, :, 2] > (115 if head else 125))
            & ((rgb[:, :, 2] - rgb[:, :, 0] > 2) | head)).astype('uint8') * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), 'uint8'))


def rings(image, head=False):
    """Find white square rings with a wood-coloured, quadrilateral opening."""
    image = normalize(image)
    mask = white_mask(image, head)
    if head:
        mask[:240] = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    if hierarchy is None:
        return found
    for i, hole in enumerate(contours):
        parent = hierarchy[0, i, 3]
        if parent < 0:
            continue
        area = cv2.contourArea(hole)
        if not (25 if head else 180) < area < (6000 if head else 65000):
            continue
        outer = contours[parent]
        outer_area = cv2.contourArea(outer)
        if not 1.7 < outer_area / area < 30:
            continue
        rect = cv2.minAreaRect(hole)
        if min(rect[1]) < 4 or max(rect[1]) / min(rect[1]) > 2.8:
            continue
        rectangularity = area / max(np.prod(rect[1]), 1)
        poly = cv2.approxPolyDP(hole, .05 * cv2.arcLength(hole, True), True)
        if rectangularity < .68 or len(poly) not in (4, 5, 6):
            continue
        hm = np.zeros(mask.shape, 'uint8')
        cv2.drawContours(hm, [hole], -1, 255, -1)
        hm = cv2.erode(hm, np.ones((3, 3), 'uint8'))
        rgb = image[hm > 0]
        if not len(rgb):
            continue
        color = np.median(rgb, axis=0)
        # A wooden opening, not the black opening of a fiducial or robot part.
        if color.mean() < 65 or (head and color[0] - color[2] < 5):
            continue
        center = np.array(rect[0])
        box = cv2.boxPoints(rect)
        axes = [(box[(k + 1) % 4] - box[k]) for k in (0, 1)]
        axes = [a / np.linalg.norm(a) for a in axes]
        pts = outer[:, 0].astype(float)
        # Limit attached gripper/table components using the opening's local axes.
        local = np.stack([(pts-center) @ axis for axis in axes], axis=1)
        lengths = [np.ptp((box-center) @ axis) for axis in axes]
        pts = pts[np.all(np.abs(local) < np.array(lengths)*2.3, axis=1)]
        if len(pts) < 4:
            continue
        # Ring plus tab has an asymmetric extent along one opening axis.
        candidates = []
        for axis in axes:
            proj = (pts - center) @ axis
            candidates.extend([(float(proj.max() + proj.min()), axis),
                               (float(-proj.min() - proj.max()), -axis)])
        asymmetry, direction = max(candidates, key=lambda v: v[0])
        depth = (pts - center) @ direction
        extent = float(depth.max())
        side = np.array([-direction[1], direction[0]])
        tip_points = pts[depth > extent - max(2, .15 * extent)]
        lateral = float(np.median((tip_points - center) @ side))
        tip = center + direction * extent + side * lateral
        # Tab grasp point is behind its tip, away from the ring body.
        tab = center + .86 * (tip - center)
        found.append(dict(center=center.tolist(), hole=box.tolist(),
                          area=float(area), outer_area=float(outer_area),
                          tab=tab.tolist(), tip=tip.tolist(),
                          direction=direction.tolist(),
                          tab_confidence=float(asymmetry / max(extent, 1)),
                          score=float(rectangularity),
                          bbox=list(cv2.boundingRect(outer))))
    return sorted(found, key=lambda v: v['area'], reverse=True)


def peg_candidates(image, square=None):
    """White base plus narrower upright; top is a quadrilateral white cap."""
    image = normalize(image)
    mask = white_mask(image)
    mask[:265] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if not 500 < area < 6500 or not .45 < w / h < 1.7 or y + h > 465:
            continue
        center = np.array([x + w / 2, y + h / 2])
        if square is not None and np.linalg.norm(center - square['center']) < max(w, h):
            continue
        # Topmost white rows form the cap. Base must be significantly wider.
        region = mask[y:y+h, x:x+w]
        widths = np.sum(region > 0, axis=1)
        cap_height = max(3, int(.14 * h))
        cap_width = float(np.median(widths[2:cap_height+2]))
        if not .12 < cap_width / w < .70:
            continue
        yy, xx = np.nonzero(region[:cap_height])
        if len(xx) < 12:
            continue
        top = np.array([x + np.median(xx), y + np.median(yy)])
        result.append(dict(top=top.tolist(), base=center.tolist(),
                           bbox=[x, y, w, h], area=float(area),
                           cap_width=cap_width, score=float(1 - abs(cap_width/w - .33))))
    return sorted(result, key=lambda p: p['score'], reverse=True)


def choose_ring(candidates, previous=None):
    if not candidates:
        return None
    if previous is None:
        return candidates[0]
    return min(candidates, key=lambda r: np.linalg.norm(np.array(r['center']) - previous))


def marker_center(image):
    """Historical left wrist fiducial face, for geometric calibration only."""
    image = normalize(image)
    mask = white_mask(image, True)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    if hierarchy is None:
        return None
    for i, c in enumerate(contours):
        a = cv2.contourArea(c)
        if hierarchy[0, i, 3] < 0 or not 180 < a < 1100:
            continue
        rect = cv2.minAreaRect(c)
        x, y = rect[0]
        if not (100 < x < 380 and 315 < y < 470):
            continue
        if min(rect[1]) < 8 or max(rect[1])/min(rect[1]) > 1.9:
            continue
        hm = np.zeros(mask.shape, 'uint8')
        cv2.drawContours(hm, [c], -1, 255, -1)
        rgb = np.median(image[hm > 0], axis=0)
        if rgb.mean() < 90 and a / np.prod(rect[1]) > .70:
            candidates.append((a, list(rect[0])))
    return max(candidates)[1] if candidates else None


def caps(image, hole_area=None):
    """Quadrilateral white horizontal caps, filtered against hole scale."""
    image = normalize(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 25, 70)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    for threshold in range(135, 221, 10):
        mask = ((gray > threshold) & (image[:, :, 2].astype(float)
                - image[:, :, 0] > 3)).astype('uint8')*255
        extra, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(extra)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not 120 < area < 20000:
            continue
        poly = cv2.approxPolyDP(c, .045 * cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        points = poly[:, 0]
        center = points.mean(axis=0)
        if center[1] > 415:
            continue
        if hole_area and not .08 < area / hole_area < 1.2:
            continue
        mask = np.zeros(gray.shape, 'uint8')
        cv2.drawContours(mask, [poly], -1, 255, -1)
        mask = cv2.erode(mask, np.ones((5, 5), 'uint8'))
        rgb = image[mask > 0].astype(float)
        if len(rgb) < 30:
            continue
        med = np.median(rgb, axis=0)
        # The cap is pale blue-white in the wrist views, with low texture.
        if med.mean() < 125 or med[2] - med[0] < 2 or np.std(rgb.mean(axis=1)) > 22:
            continue
        rect = cv2.minAreaRect(poly)
        if min(rect[1]) < 5 or max(rect[1])/min(rect[1]) > 2.5:
            continue
        if any(np.linalg.norm(center - v['center']) < 8 for v in out):
            continue
        out.append(dict(center=center.tolist(), polygon=points.tolist(), area=float(area)))
    return out


def tab_at_jaws(image, ring):
    """Measure the pale tab crossing the lower, central wrist gripper throat."""
    mask = white_mask(normalize(image))
    center = ring['center'][0]
    row = np.mean(mask[410:435] > 0, axis=0) > .65
    edges = np.diff(np.r_[False, row, False].astype(int))
    runs = [(int(a), int(b)) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
            if 35 < b-a < 190 and abs((a+b)/2-center) < 90]
    if not runs:
        return None
    a, b = min(runs, key=lambda ab: abs((ab[0]+ab[1])/2-center))
    return dict(center=[(a+b)/2, 422.5], width=b-a, interval=[a, b])
