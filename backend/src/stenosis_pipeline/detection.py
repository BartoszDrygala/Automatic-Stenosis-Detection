import cv2
import numpy as np

#***************************************************************
# Candidate detections
#***************************************************************

def detect_stenosis_global(path, dist, smooth=15, drop_thresh=0.35):
    values = np.array([dist[y, x] for (x, y) in path])

    if len(values) < 30:
        return [], values, None

    values_smooth = np.convolve(values, np.ones(smooth)/smooth, mode='same')

    max_val = np.max(values_smooth)
    if max_val < 1e-6:
        return [], values, None

    drop = (max_val - values_smooth) / max_val

    idx = np.where(drop > drop_thresh)[0]
    idx = [i for i in idx if 20 < i < len(values) - 20]

    return idx, values_smooth, drop


def detect_stenosis_local_simple(path, dist, window=15, drop_thresh=0.3):
    values = np.array([dist[y, x] for (x, y) in path])

    if len(values) < window * 2:
        return [], values, None

    values = np.convolve(values, np.ones(5)/5, mode='same')

    idxs = []

    for i in range(window, len(values) - window):
        local = values[i-window:i+window]
        center = values[i]

        ref = max(
            np.median(local[:window]),
            np.median(local[window:])
        )

        drop = (ref - center) / (ref + 1e-6)

        if drop > drop_thresh:
            idxs.append(i)

    return idxs, values, None



#***************************************************************
# FP cleanup
#***************************************************************

def contrast_filter(idx, values, window=10, min_drop=0.25):
    left = values[max(0, idx-window):idx]
    right = values[idx:min(len(values), idx+window)]

    if len(left) < 3 or len(right) < 3:
        return False

    ref = max(np.median(left), np.median(right))
    val = values[idx]

    drop = (ref - val) / (ref + 1e-6)

    return drop > min_drop


def not_tapering(idx, values, window=15):
    left = values[max(0, idx-window):idx]
    right = values[idx:min(len(values), idx+window)]

    if len(left) < 5 or len(right) < 5:
        return False

    return not (np.mean(right) < np.mean(left) * 0.85)


def refine_to_minimum(idx, values, window=10):
    left = max(0, idx - window)
    right = min(len(values), idx + window)

    local = values[left:right]

    if len(local) == 0:
        return idx

    return left + np.argmin(local)


def compute_stenosis_severity(idx, values, window=30):
    val = values[idx]

    left = values[max(0, idx-window):idx]
    right = values[idx:min(len(values), idx+window)]

    if len(left) < 5 or len(right) < 5:
        return 0.0

    ref = max(np.max(left), np.max(right))

    if ref < 1e-6:
        return 0.0

    return (ref - val) / ref


def is_wide_valley(idx, values, width=5):
    left = max(0, idx - width)
    right = min(len(values), idx + width)

    segment = values[left:right]

    return np.mean(segment) < values[idx] * 1.2


def split_path(path, segment_length=90, overlap=30):
    segments = []
    step = segment_length - overlap

    for i in range(0, len(path), step):
        seg = path[i:i+segment_length]

        if len(seg) > 20:
            segments.append((seg, i))

    return segments


def detect_stenosis_combined(path, dist):
    g_idx, _, g_score = detect_stenosis_global(path, dist)
    l_idx, _, _ = detect_stenosis_local_simple(path, dist)

    g_set = set(g_idx)
    l_set = set(l_idx)

    candidates = sorted(list(g_set.union(l_set)))

    values = np.array([dist[y, x] for (x, y) in path])

    def get_score(idx):
        score = 0

        if idx in g_set and g_score is not None and idx < len(g_score):
            score += g_score[idx]

        if idx in l_set:
            score += 0.3

        if idx in g_set and idx in l_set:
            score += 0.5

        return score

    candidates = sorted(candidates, key=get_score, reverse=True)

    selected = []

    for idx in candidates:
        if idx >= len(values):
            continue

        if not not_tapering(idx, values):
            continue

        if not contrast_filter(idx, values):
            continue

        refined_idx = refine_to_minimum(idx, values)
        severity = compute_stenosis_severity(refined_idx, values)

        if not is_wide_valley(refined_idx, values):
            continue

        if severity < 0.35:
            continue

        if any(abs(refined_idx - s[0]) < 20 for s in selected):
            continue

        selected.append((refined_idx, severity, idx))

        if len(selected) >= 2:
            break

    return selected, g_set, l_set


#***************************************************************
# Region reconstruction
#***************************************************************

def extract_full_stenosis_region(idx, values, recovery_ratio=0.80, local_window=30, max_expand=40): 
    """
    Better stenosis region extraction.

    Instead of extracting only the narrowest point,
    this expands until vessel width recovers to
    ~90% of nearby healthy vessel width.

    Result:
    healthy → narrowing → stenosis → recovery
    """

    if idx >= len(values):
        return idx, idx

    # local healthy reference
    left_zone = values[max(0, idx - local_window):idx]
    right_zone = values[idx:min(len(values), idx + local_window)]

    if len(left_zone) < 3 or len(right_zone) < 3:
        return max(0, idx - 10), min(len(values) - 1, idx + 10)

    left_ref = np.max(left_zone)
    right_ref = np.max(right_zone)

    target_left = left_ref * recovery_ratio
    target_right = right_ref * recovery_ratio

    left = idx
    right = idx

    # expand left until healthy enough
    for i in range(idx, max(idx - max_expand, 0), -1):
        left = i
        if values[i] >= target_left:
            break

    # expand right until healthy enough
    for i in range(idx, min(idx + max_expand, len(values))):
        right = i
        if values[i] >= target_right:
            break

    # safety minimum size
    if right - left < 8:
        left = max(0, idx - 8)
        right = min(len(values) - 1, idx + 8)

    return left, right


def build_exact_stenosis_mask(region_pts, mask):
    """
    Build region that aligns to vessel walls.

    Better than fixed tiny dilation.
    Expands inside vessel mask only.
    """

    region_mask = np.zeros_like(mask, dtype=np.uint8)

    for (x, y) in region_pts:
        region_mask[y, x] = 255

    kernel = np.ones((3, 3), np.uint8)

    # stronger adaptive expansion
    for _ in range(12):
        dilated = cv2.dilate(region_mask, kernel, iterations=1)

        # constrain strictly to vessel mask
        dilated = cv2.bitwise_and(dilated, mask)

        # stop if stable
        if np.array_equal(dilated, region_mask):
            break

        region_mask = dilated

    return region_mask


def get_main_contour(region_mask):
    contours, _ = cv2.findContours(region_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)


#***************************************************************
# Main logic
#***************************************************************

def contours_close_or_overlap(c1, c2, distance_threshold=20):
    """
    Merge stenoses if contours:
    - overlap
    - touch
    - are spatially very close
    """

    x1, y1, w1, h1 = cv2.boundingRect(c1)
    x2, y2, w2, h2 = cv2.boundingRect(c2)

    # overlap check
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)

    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)

    if right >= left and bottom >= top:
        return True

    # distance between rectangles
    dx = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
    dy = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))

    dist = np.sqrt(dx * dx + dy * dy)

    return dist <= distance_threshold


def merge_contours(c1, c2, vessel_mask):
    merged_mask = np.zeros_like(vessel_mask, dtype=np.uint8)

    cv2.drawContours(merged_mask, [c1], -1, 255, thickness=-1)
    cv2.drawContours(merged_mask, [c2], -1, 255, thickness=-1)

    merged_mask = cv2.bitwise_and(merged_mask, vessel_mask)

    kernel = np.ones((3, 3), np.uint8)
    merged_mask = cv2.dilate(merged_mask, kernel, iterations=1)
    merged_mask = cv2.bitwise_and(merged_mask, vessel_mask)

    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return c1

    return max(contours, key=cv2.contourArea)


def detect_all_stenosis(paths, dist, mask):
    all_detections = []

    for path in paths:
        if len(path) > 120:
            segments = split_path(path, overlap = 40)
        else:
            segments = [(path, 0)]

        # compute once per path
        values = np.array([
            dist[y, x] for (x, y) in path
        ])

        for seg, offset in segments:
            # detection logic unchanged
            selected, g_set, l_set = detect_stenosis_combined(seg, dist)

            for refined_idx, severity, original_idx in selected:
                global_idx = refined_idx + offset
                original_global_idx = original_idx + offset

                if global_idx >= len(path):
                    continue

                ####################################################
                # NEW BETTER REGION EXTRACTION
                ####################################################

                left, right = extract_full_stenosis_region(global_idx, values, recovery_ratio=0.80,local_window=30, max_expand=20)

                region_pts = path[left:right + 1]

                ####################################################
                # BETTER VESSEL-ALIGNED CONTOUR
                ####################################################

                region_mask = build_exact_stenosis_mask(region_pts, mask)
                contour = get_main_contour(region_mask)

                if contour is None:
                    continue

                all_detections.append({
                    "path": path,
                    "idx": global_idx,
                    "orig_idx": original_global_idx,
                    "severity": severity,
                    "contour": contour,
                    "left": left,
                    "right": right,
                    "g_set": g_set,
                    "l_set": l_set
                })

    ############################################################
    # FINAL MERGING
    #
    # merge nearby lesions into one larger lesion
    ############################################################

    final = []

    all_detections = sorted(all_detections, key=lambda x: x["severity"], reverse=True)
    #return all_detections
    for det in all_detections:
        merged = False

        for existing in final:
            if contours_close_or_overlap(det["contour"], existing["contour"], distance_threshold=20):
                # merge contours instead of deleting
                existing["contour"] = merge_contours(existing["contour"], det["contour"], mask)
                existing["left"] = min(existing["left"], det["left"])
                existing["right"] = max(existing["right"], det["right"])
                existing["severity"] = max(existing["severity"],det["severity"])

                merged = True
                break

        if not merged:
            final.append(det)

    return final