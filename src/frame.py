import cv2

def process_frame(img, extractor, matcher, display, K, W, H):
    img = cv2.resize(img, (W, H))
    feats = extractor.extract(img)

    vis, n_good, n_kpts, pts_prev, pts_curr = matcher.match_and_draw(img, feats)

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

    if pts_prev is not None or pts_curr is not None:
        # Compute the Essensial matrix 
        E, inlier_mask = cv2.findEssentialMat(pts_prev, pts_curr, cameraMatrix=K,
                                    method=cv2.USAC_MAGSAC,
                                    prob=0.999,   
                                    threshold=1.0)
    
        if E is not None and inlier_mask is not None:
            print(f"Essential matrix: {E}")
            
            # Recover the relative camera pose
            valid_points, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_curr, cameraMatrix=K, mask=inlier_mask)

            print(f"Rotation matrix: {R}")
            print(f"translation vector: {t}")


    # Triangulation will be here

    display.paint(vis)