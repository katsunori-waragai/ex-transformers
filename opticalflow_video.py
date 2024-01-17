import cv2

from opticalflow2 import (
    compute_grid_indices,
    compute_optical_flow,
    model,
    normalize,
    visualize_flow,
)

if __name__ == "__main__":
    movie_name = "hand.mp4"
    movie_name = "nod.webm"
    cap = cv2.VideoCapture(movie_name)
    counter = -1
    r1, im1 = cap.read()
    r2, im2 = cap.read()
    if not r2:
        exit()

    counter += 1
    out_movie = "opticalflow_out.mp4"

    H, W = im1.shape[:2]

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(str(out_movie), fourcc, 20.0, (W, H))

    while True:
        if not r2:
            break
        if counter > 50:
            break

        grid_indices = compute_grid_indices(im1.shape)
        t0 = cv2.getTickCount()
        flow = compute_optical_flow(model, normalize(im1), normalize(im2), grid_indices)
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used=} {im1.shape=}")
        bgr = visualize_flow(flow[0])
        writer.write(bgr)
        r1, im1 = cap.read()
        r2, im2 = cap.read()
        counter += 1

    writer.release()
