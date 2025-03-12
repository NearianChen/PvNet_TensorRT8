import os
import cv2
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

# 配置路径
rosbag_path = "data/rosbag/k29pot_v2.bag"  # 替换为你的ROSbag文件路径
output_dir = "data/test_0117"      # 替换为你希望保存图片的路径
topic_name = "/hdas/camera_head/left_raw/image_raw_color/compressed"

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 初始化CvBridge
bridge = CvBridge()

# 打开rosbag文件
with rosbag.Bag(rosbag_path, 'r') as bag:
    count = 0
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if topic == topic_name:
            try:
                # 将CompressedImage消息转换为OpenCV图像
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

                # 构造文件名
                image_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")

                # 保存图片到本地
                cv2.imwrite(image_filename, cv_image)
                print(f"Saved {image_filename}")
                count += 1
            except Exception as e:
                print(f"Failed to process message: {e}")

print(f"Finished extracting images. Total images saved: {count}")
