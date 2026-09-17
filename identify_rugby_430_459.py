import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
ITEM_ID = "2ea4bb57-a358-4349-8aaf-204ded0772ab"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
POINT = (430, 459)


def main():
    result_data = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    image_bytes = requests.get(result_data["clean_image_url"]).content

    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    binary_mask = (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255
    h_img, w_img = binary_mask.shape

    x, y = POINT
    print(f"Image dimensions: {w_img} x {h_img}")
    print(f"Point {POINT} → ({x/w_img*100:.1f}%, {y/h_img*100:.1f}%) of image")

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    lowest_idx = np.argmax(main_contour[:, 0, 1])
    lowest_pt = tuple(int(v) for v in main_contour[lowest_idx][0])
    dist_to_lowest = np.hypot(x - lowest_pt[0], y - lowest_pt[1])
    print(f"\nActual lowest contour point: {lowest_pt} "
          f"({lowest_pt[0]/w_img*100:.1f}%, {lowest_pt[1]/h_img*100:.1f}%)")
    print(f"Distance from {POINT} to lowest point: {dist_to_lowest:.1f}px")


if __name__ == "__main__":
    main()