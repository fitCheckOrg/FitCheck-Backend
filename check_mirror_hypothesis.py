import io
import requests
from PIL import Image, ImageDraw

ITEM_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

LEFT_NOISE_POINT = (88, 217)      # correctly filtered on LEFT
RIGHT_PROBLEM_POINT = (379, 221)  # still winning incorrectly on RIGHT

result_data = requests.get(
    f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
).json()["data"]
image_bytes = requests.get(result_data["clean_image_url"]).content
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
w, h = image.size

draw = ImageDraw.Draw(image)
for pt, color, label in [(LEFT_NOISE_POINT, (255,0,0), "LEFT_noise"),
                            (RIGHT_PROBLEM_POINT, (0,200,255), "RIGHT_problem")]:
    x, y = pt
    draw.ellipse([x-10,y-10,x+10,y+10], outline=color, width=4)
    draw.text((x+12,y-8), label, fill=color)

image.save("mirror_hypothesis_check.png")
print(f"Image size: {w}x{h}")
print(f"LEFT point: {LEFT_NOISE_POINT}  mirror would be ~({w-LEFT_NOISE_POINT[0]}, {LEFT_NOISE_POINT[1]})")
print(f"RIGHT point: {RIGHT_PROBLEM_POINT}")
print("Saved: mirror_hypothesis_check.png")