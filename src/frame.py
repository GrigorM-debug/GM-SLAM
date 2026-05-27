import cv2

def process_frame(img, W, H, extractor, matcher, display):

    feats = extractor.extract(img)

    vis, n_good, n_kpts = matcher.match_and_draw(img, feats)

    label = f"Features: {n_kpts}  Matches: {n_good}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 3

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = (vis.shape[1] - text_w) // 2
    y = 20 + text_h

    cv2.putText(
        vis,
        label,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    display.paint(vis)